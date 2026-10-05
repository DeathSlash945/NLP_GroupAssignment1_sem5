"""
nlp_pipeline.py
================
Shared backend for Question 4 (Integrated Background Editor), imported by BOTH the
demo notebook (Group_Assignment_Q4.ipynb) and the Streamlit app (streamlit_app.py).

Blocks:
  A. Q1 REUSED   -- trigram word-LM + Viterbi segmentation, trigram HMM POS tagger.
  B. Q3 REUSED   -- vocabulary/unigram/bigram models, Method A / Method B candidates,
                    non-word / real-word correction.
  C. Q4 NEW      -- merged-token typing simulator, PCFG parser + tagset reconciliation,
                    decision rule.
  D. ORCHESTRATION -- LiveEditorSession, end-of-passage analysis, Speed-Demon benchmark.

Design choices:
  MERGE_PROB (p) = 0.08   a typist misses the spacebar about once per 12 word boundaries:
      frequent enough that segmentation has real work, rare enough not to drown the
      spelling and grammar layers.
  GRAMMAR_TRIGGER_N = 15  the grammar/real-word check is heavier than the per-token vocab
      lookups, so it runs every 15 tokens (about a sentence). Smaller N catches errors
      sooner but is noisier; larger N delays detection.
  ADD_K = 0.5             matches the smoothing used for Q1's segmentation LM so
      perplexities are on a comparable scale.
"""

import re
import math
import time
import random
import string
from collections import defaultdict, Counter

random.seed(42)

BOS = "<s>"
EOS = "</s>"

MERGE_PROB = 0.08
GRAMMAR_TRIGGER_N = 15
ADD_K = 0.5
PERPLEXITY_ALERT_MULTIPLIER = 2.5   # window flagged if perplexity > this x avg training perplexity
REALWORD_THRESHOLD = 2.0            # matches Q3's correct_realword threshold


# ======================================================================
# A. Q1 REUSED
# ======================================================================

UNK_LOGPROB_PER_CHAR = -4.0


def unk_penalty(word):
    return UNK_LOGPROB_PER_CHAR * len(word)


class TrigramLM:
    """Trigram model over words, add-k smoothed with back-off to bigram/unigram."""

    def __init__(self, k=ADD_K):
        self.k = k
        self.uni = Counter()
        self.bi = Counter()
        self.tri = Counter()
        self.vocab = set()
        self.total = 0

    def train(self, sentences_of_words):
        for words in sentences_of_words:
            seq = [BOS, BOS] + list(words) + [EOS]
            for w in words:
                self.vocab.add(w)
            for i in range(len(seq)):
                self.uni[seq[i]] += 1
                self.total += 1
                if i >= 1:
                    self.bi[(seq[i - 1], seq[i])] += 1
                if i >= 2:
                    self.tri[(seq[i - 2], seq[i - 1], seq[i])] += 1
        self.V = max(len(self.vocab), 1)

    def logprob(self, w, u=None, v=None):
        """log P(w | v, u) with back-off: trigram -> bigram -> unigram."""
        k = self.k
        if u is not None and v is not None:
            num = self.tri[(v, u, w)] + k
            den = self.bi[(v, u)] + k * self.V
            if den > 0:
                return math.log(num / den)
        if u is not None:
            num = self.bi[(u, w)] + k
            den = self.uni[u] + k * self.V
            if den > 0:
                return math.log(num / den)
        num = self.uni[w] + k
        den = self.total + k * self.V
        return math.log(num / max(den, 1e-12))

    def sentence_logprob(self, words):
        seq = [BOS, BOS] + list(words) + [EOS]
        lp = 0.0
        for i in range(2, len(seq)):
            lp += self.logprob(seq[i], seq[i - 1], seq[i - 2])
        return lp

    def perplexity(self, words):
        n = max(len(words) + 1, 1)  # +1 for EOS
        return math.exp(-self.sentence_logprob(words) / n)


class BigramLM:
    """Add-k smoothed bigram LM (the 'bigram model' Q4 trains and reports)."""

    def __init__(self, k=ADD_K):
        self.k = k
        self.uni = Counter()
        self.bi = Counter()
        self.vocab = set()
        self.total = 0

    def train(self, sentences_of_words):
        for words in sentences_of_words:
            seq = [BOS] + list(words) + [EOS]
            for w in words:
                self.vocab.add(w)
            for i in range(len(seq)):
                self.uni[seq[i]] += 1
                self.total += 1
                if i >= 1:
                    self.bi[(seq[i - 1], seq[i])] += 1
        self.V = max(len(self.vocab), 1)

    def logprob(self, w, u=None):
        k = self.k
        if u is not None:
            num = self.bi[(u, w)] + k
            den = self.uni[u] + k * self.V
            return math.log(num / den)
        num = self.uni[w] + k
        den = self.total + k * self.V
        return math.log(num / max(den, 1e-12))

    def sentence_logprob(self, words):
        seq = [BOS] + list(words) + [EOS]
        lp = 0.0
        for i in range(1, len(seq)):
            lp += self.logprob(seq[i], seq[i - 1])
        return lp

    def perplexity(self, words):
        n = max(len(words) + 1, 1)
        return math.exp(-self.sentence_logprob(words) / n)


def viterbi_segment(chars, lm, vocab, max_word_len=15):
    """Second-order Viterbi word segmentation (Q1's decoder).
    Always returns (words, best_total_score)."""
    n = len(chars)
    dp = [dict() for _ in range(n + 1)]
    dp[0][BOS] = (0.0, BOS, None)

    for j in range(1, n + 1):
        best_for_word = {}
        start = max(0, j - max_word_len)
        for i in range(start, j):
            w = chars[i:j]
            if not dp[i]:
                continue
            in_vocab = w in vocab
            for u, (score_i, v, _back) in dp[i].items():
                lp = lm.logprob(w, u, v)
                if not in_vocab:
                    lp += unk_penalty(w)
                cand = score_i + lp
                if w not in best_for_word or cand > best_for_word[w][0]:
                    best_for_word[w] = (cand, u, i)
        dp[j] = best_for_word

    if not dp[n]:
        return list(chars), float("-inf")

    best_word = max(dp[n], key=lambda w: dp[n][w][0])
    words = []
    j, w = n, best_word
    while j > 0:
        sc, u, i = dp[j][w]
        words.append(chars[i:j])
        j, w = i, u
    words.reverse()
    return words, dp[n][best_word][0]


def segmentation_score_as_one_word(token, lm, vocab):
    """Score of treating `token` as one (possibly OOV) word, for the SEGMENT-ALERT test."""
    lp = lm.logprob(token, BOS, BOS)
    if token not in vocab:
        lp += unk_penalty(token)
    return lp


def greedy_longest_match_segment(chars, vocab, max_word_len=15):
    n = len(chars)
    i, words = 0, []
    while i < n:
        matched = None
        for j in range(min(n, i + max_word_len), i, -1):
            if chars[i:j] in vocab:
                matched = chars[i:j]
                i = j
                break
        if matched is None:
            matched = chars[i]
            i += 1
        words.append(matched)
    return words


_BROWN_TO_UNIVERSAL_PREFIXES = [
    ("NP", "NOUN"), ("NR", "NOUN"), ("NN", "NOUN"),
    ("VB", "VERB"), ("DO", "VERB"), ("HV", "VERB"), ("BE", "VERB"), ("MD", "VERB"),
    ("JJ", "ADJ"),
    ("QL", "ADV"), ("RB", "ADV"), ("RN", "ADV"), ("RP", "PRT"), ("WRB", "ADV"),
    ("IN", "ADP"),
    ("CC", "CONJ"), ("CS", "CONJ"),
    ("WDT", "DET"), ("AT", "DET"), ("DT", "DET"), ("AP", "DET"), ("AB", "DET"),
    ("CD", "NUM"), ("OD", "NUM"),
    ("WPS", "PRON"), ("WPO", "PRON"), ("WP", "PRON"), ("PP", "PRON"), ("PN", "PRON"),
    ("TO", "PRT"),
    ("UH", "X"), ("FW", "X"), ("EX", "DET"), ("NIL", "X"),
]


def brown_tag_to_universal(tag):
    """Brown/PTB -> Universal (coarse) tag mapping (Q1's function; also used for
    the Q1 <-> PTB tagset reconciliation in Q4)."""
    t = tag.split('-')[0].split('+')[0].rstrip('*').rstrip('$')
    if not t or not t[0].isalpha():
        return "."
    for prefix, universal in _BROWN_TO_UNIVERSAL_PREFIXES:
        if t.startswith(prefix):
            return universal
    return "X"


class TrigramHMM:
    """Trigram HMM POS tagger with beam-pruned Viterbi decoding (Q1's tagger)."""

    def __init__(self, k_trans=0.1, k_emit=0.1):
        self.k_trans, self.k_emit = k_trans, k_emit
        self.trans_tri = Counter()
        self.trans_bi = Counter()
        self.emit = Counter()
        self.tag_count = Counter()
        self.word_count = Counter()
        self.tags = set()

    def train(self, tagged_sentences):
        for sent in tagged_sentences:
            tags = [BOS, BOS] + [t for _, t in sent]
            for _, t in sent:
                self.tags.add(t)
            for i in range(2, len(tags)):
                self.trans_tri[(tags[i - 2], tags[i - 1], tags[i])] += 1
                self.trans_bi[(tags[i - 2], tags[i - 1])] += 1
            for w, t in sent:
                self.emit[(t, w)] += 1
                self.tag_count[t] += 1
                self.word_count[w] += 1
        self.T = max(len(self.tags), 1)
        self.tag_list = sorted(self.tags)

    def trans_logprob(self, t, t1, t2):
        num = self.trans_tri[(t2, t1, t)] + self.k_trans
        den = self.trans_bi[(t2, t1)] + self.k_trans * self.T
        return math.log(num / den)

    def emit_logprob(self, w, t):
        if self.word_count[w] == 0:
            return math.log(self.k_emit / (self.tag_count[t] + self.k_emit * self.T))
        num = self.emit[(t, w)] + self.k_emit
        den = self.tag_count[t] + self.k_emit * len(self.word_count)
        return math.log(num / den)

    def viterbi_tag(self, words, beam_size=100):
        n = len(words)
        if n == 0:
            return []
        tags = self.tag_list
        dp = [dict() for _ in range(n + 1)]
        dp[0][(BOS, BOS)] = (0.0, None)
        for i in range(1, n + 1):
            w = words[i - 1]
            emit_cache = {t: self.emit_logprob(w, t) for t in tags}
            prev_states = dp[i - 1]
            if beam_size is not None and len(prev_states) > beam_size:
                prev_states = dict(sorted(prev_states.items(), key=lambda kv: kv[1][0],
                                          reverse=True)[:beam_size])
            for (t_im2, t_im1), (score, _prev) in prev_states.items():
                for t in tags:
                    lp = score + self.trans_logprob(t, t_im1, t_im2) + emit_cache[t]
                    state = (t_im1, t)
                    if state not in dp[i] or lp > dp[i][state][0]:
                        dp[i][state] = (lp, (t_im2, t_im1))
            if beam_size is not None and len(dp[i]) > beam_size:
                dp[i] = dict(sorted(dp[i].items(), key=lambda kv: kv[1][0],
                                    reverse=True)[:beam_size])
        if not dp[n]:
            return [self.tag_list[0]] * n
        best_state = max(dp[n], key=lambda s: dp[n][s][0])
        tags_out, state, i = [], best_state, n
        while i > 0:
            score, prev_state = dp[i][state]
            tags_out.append(state[1])
            state = prev_state
            i -= 1
        tags_out.reverse()
        return tags_out


def load_english_data():
    """Brown tagged sentences (Universal coarse tags); embedded fallback if unavailable."""
    try:
        import nltk
        from nltk.corpus import brown
        try:
            brown.tagged_sents()
        except LookupError:
            nltk.download('brown')
        sents = brown.tagged_sents()
        data = [[(w.lower(), brown_tag_to_universal(t)) for w, t in s] for s in sents]
        print(f"[english] loaded {len(data)} sentences from NLTK Brown corpus")
        return data
    except Exception as e:
        print(f"[english] NLTK/Brown unavailable ({e}); using embedded fallback corpus")
        return ENGLISH_FALLBACK * 60


ENGLISH_FALLBACK = [
    [("the", "DET"), ("quick", "ADJ"), ("brown", "ADJ"), ("fox", "NOUN"), ("jumps", "VERB"),
     ("over", "ADP"), ("the", "DET"), ("lazy", "ADJ"), ("dog", "NOUN")],
    [("the", "DET"), ("dog", "NOUN"), ("barks", "VERB"), ("at", "ADP"), ("the", "DET"), ("cat", "NOUN")],
    [("a", "DET"), ("quick", "ADJ"), ("cat", "NOUN"), ("runs", "VERB"), ("over", "ADP"),
     ("the", "DET"), ("lazy", "ADJ"), ("fox", "NOUN")],
    [("she", "PRON"), ("eats", "VERB"), ("a", "DET"), ("green", "ADJ"), ("salad", "NOUN")],
    [("the", "DET"), ("man", "NOUN"), ("saw", "VERB"), ("the", "DET"), ("dog", "NOUN"),
     ("with", "ADP"), ("a", "DET"), ("telescope", "NOUN")],
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("on", "ADP"), ("the", "DET"), ("mat", "NOUN")],
    [("i", "PRON"), ("would", "VERB"), ("like", "VERB"), ("to", "PRT"), ("see", "VERB"), ("the", "DET"), ("world", "NOUN")],
    [("please", "VERB"), ("meet", "VERB"), ("me", "PRON"), ("at", "ADP"), ("the", "DET"), ("station", "NOUN")],
    [("this", "DET"), ("is", "VERB"), ("a", "DET"), ("test", "NOUN"), ("sentence", "NOUN")],
    [("i", "PRON"), ("have", "VERB"), ("a", "DET"), ("good", "ADJ"), ("feeling", "NOUN"), ("about", "ADP"), ("this", "DET")],
]


def most_frequent_tag_baseline(train_tagged_sentences):
    counts = defaultdict(Counter)
    overall = Counter()
    for sent in train_tagged_sentences:
        for w, t in sent:
            counts[w][t] += 1
            overall[t] += 1
    word2tag = {w: c.most_common(1)[0][0] for w, c in counts.items()}
    default_tag = overall.most_common(1)[0][0]
    return word2tag, default_tag


# ======================================================================
# B. Q3 REUSED
# ======================================================================

alphabet = string.ascii_lowercase


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in alphabet]
    inserts = [L + c + R for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def one_char_deletes(word):
    if len(word) <= 1:
        return {""}
    return {word[:i] + word[i + 1:] for i in range(len(word))}


class SpellingModel:
    """Q3's vocabulary, unigram/bigram counts and both candidate-generation methods."""

    def __init__(self, k_bigram=1):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.total_word_count = 0
        self.symspell_dict = defaultdict(set)
        self.k_bigram = k_bigram

    def train(self, sentences_of_words):
        for s in sentences_of_words:
            self.unigram_counts.update(s)
        self.vocab = set(self.unigram_counts.keys())
        self.total_word_count = sum(self.unigram_counts.values())
        for s in sentences_of_words:
            for w1, w2 in zip(s, s[1:]):
                self.bigram_counts[w1][w2] += 1
        self.VOCAB_SIZE = len(self.vocab)
        for w in self.vocab:                      # Method B preprocessing (SymSpell)
            self.symspell_dict[w].add(w)
            for d in one_char_deletes(w):
                self.symspell_dict[d].add(w)

    def unigram_prob(self, word):
        return self.unigram_counts.get(word, 0) / max(self.total_word_count, 1)

    def bigram_prob(self, w1, w2):
        k = self.k_bigram
        denom = self.unigram_counts.get(w1, 0) + k * self.VOCAB_SIZE
        return (self.bigram_counts[w1][w2] + k) / denom

    def method_a_candidates(self, word):
        return {w for w in edits1(word) if w in self.vocab}

    def method_b_candidates(self, word):
        candidates = set(self.symspell_dict.get(word, set()))
        for d in one_char_deletes(word):
            candidates |= self.symspell_dict.get(d, set())
        candidates.discard('')
        return candidates

    def correct_nonword(self, word, method='b'):
        # Method B is used live by default: query cost is O(word_len) vs Method A's
        # O(word_len * 26), which matters when checks run on every typed token.
        candidates = set()
        if method in ('a', 'both'):
            candidates |= self.method_a_candidates(word)
        if method in ('b', 'both'):
            candidates |= self.method_b_candidates(word)
        if not candidates:
            return word, False
        best = max(candidates, key=lambda w: self.unigram_counts[w])
        return best, (best != word)

    def correct_realword(self, prev_word, word, next_word=None, method='both', threshold=REALWORD_THRESHOLD):
        if word not in self.vocab:
            return word, False
        candidates = set()
        if method in ('a', 'both'):
            candidates |= self.method_a_candidates(word)
        if method in ('b', 'both'):
            candidates |= self.method_b_candidates(word)
        candidates.discard(word)
        if not candidates:
            return word, False

        def phrase_score(w):
            score = 1e-12
            if prev_word is not None:
                score *= self.bigram_prob(prev_word, w)
            if next_word is not None:
                score *= self.bigram_prob(w, next_word)
            return score

        best_word, best_score = word, phrase_score(word)
        for c in candidates:
            c_score = phrase_score(c)
            if c_score > best_score * threshold:
                best_score, best_word = c_score, c
        return best_word, (best_word != word)


def apply_random_edit(word):
    if len(word) < 2:
        return word
    op = random.choice(['delete', 'insert', 'substitute', 'transpose'])
    i = random.randrange(len(word))
    if op == 'delete':
        return word[:i] + word[i + 1:]
    elif op == 'insert':
        c = random.choice(alphabet)
        return word[:i] + c + word[i:]
    elif op == 'substitute':
        c = random.choice(alphabet)
        return word[:i] + c + word[i + 1:]
    else:
        j = min(i + 1, len(word) - 1)
        chars = list(word)
        chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)


# ======================================================================
# C. Q4 NEW
# ======================================================================

def simulate_fast_typing_merges(words, p=MERGE_PROB, rng=None):
    """Drop the space between consecutive words with probability p."""
    rng = rng or random
    if not words:
        return []
    out = [words[0]]
    for w in words[1:]:
        if rng.random() < p:
            out[-1] = out[-1] + w
        else:
            out.append(w)
    return out


# ---- PCFG constituency parser (Part 2) --------------------------------
# One backend for both the real Penn Treebank grammar and the offline fallback:
# an indexed PCFG (binary, unary, lexical rules) parsed by a from-scratch CKY.

class MiniTree:
    __slots__ = ("_label", "children")

    def __init__(self, label, children):
        self._label = label
        self.children = children

    def label(self):
        return self._label

    def is_preterminal(self):
        return len(self.children) == 1 and isinstance(self.children[0], str)

    def height(self):
        if self.is_preterminal():
            return 2
        return 1 + max(c.height() for c in self.children)

    def leaves(self):
        if self.is_preterminal():
            return [self.children[0]]
        out = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def subtrees(self, filter_fn=None):
        if filter_fn is None or filter_fn(self):
            yield self
        if not self.is_preterminal():
            for c in self.children:
                yield from c.subtrees(filter_fn)

    def __repr__(self):
        if self.is_preterminal():
            return f"({self._label} {self.children[0]})"
        return f"({self._label} {' '.join(repr(c) for c in self.children)})"


UNK = "<unk>"


class MiniPCFG:
    """CNF-ish PCFG with binary, unary and lexical rules, indexed for fast CKY.
    Used for BOTH the real Penn Treebank grammar and the offline fallback."""

    def __init__(self):
        self.binary_by_children = defaultdict(list)  # (B, C) -> [(lhs, logprob)]
        self.unary_by_child = defaultdict(list)      # B -> [(lhs, logprob)]
        self.lexical_by_word = defaultdict(list)     # word -> [(lhs, logprob)]
        self.vocab = set()
        self.start = "S"

    def add_binary(self, lhs, b, c, prob):
        self.binary_by_children[(b, c)].append((lhs, math.log(prob)))

    def add_unary(self, lhs, b, prob):
        self.unary_by_child[b].append((lhs, math.log(prob)))

    def add_lexical(self, lhs, word, prob):
        self.lexical_by_word[word].append((lhs, math.log(prob)))
        self.vocab.add(word)

    def preterminals_for(self, word):
        return self.lexical_by_word.get(word, [])


def pcfg_from_nltk(nltk_grammar, start):
    g = MiniPCFG()
    g.start = start
    for prod in nltk_grammar.productions():
        lhs, rhs, p = prod.lhs().symbol(), prod.rhs(), prod.prob()
        if p <= 0:
            continue
        if len(rhs) == 2:
            g.add_binary(lhs, rhs[0].symbol(), rhs[1].symbol(), p)
        elif len(rhs) == 1:
            if isinstance(rhs[0], str):
                g.add_lexical(lhs, rhs[0], p)
            else:
                g.add_unary(lhs, rhs[0].symbol(), p)
    return g



_TOY_CNF_RULES = {
    "binary": {
        "S": [(("NP", "VP"), 1.0)],
        "NP": [(("DT", "NN"), 0.35), (("DT", "NNJ"), 0.25), (("NP", "PP"), 0.1)],
        "NNJ": [(("JJ", "NN"), 1.0)],
        "VP": [(("VBZ", "NP"), 0.25), (("VBD", "PP"), 0.15), (("VBP", "NP"), 0.2),
               (("VBZ", "PP"), 0.1), (("VBD", "NPP"), 0.15)],
        "NPP": [(("NP", "PP"), 1.0)],
        "PP": [(("IN", "NP"), 1.0)],
    },
    "lexical": {
        "DT": [("the", 0.5), ("a", 0.5)],
        "JJ": [("quick", 0.2), ("lazy", 0.2), ("green", 0.2), ("brown", 0.2), ("good", 0.2)],
        "NN": [("fox", 0.15), ("dog", 0.15), ("cat", 0.15), ("mat", 0.1), ("salad", 0.1),
               ("station", 0.1), ("world", 0.1), ("feeling", 0.05), ("sentence", 0.1)],
        "VBZ": [("jumps", 0.5), ("sits", 0.5)],
        "VBD": [("sat", 0.5), ("saw", 0.5)],
        "VBP": [("eat", 0.3), ("eats", 0.3), ("see", 0.2), ("meet", 0.2)],
        "IN": [("on", 0.5), ("over", 0.5)],
        "PRP": [("she", 0.5), ("it", 0.5)],
        "NP": [("she", 0.15), ("it", 0.15)],
    },
}


def _build_fallback_mini_pcfg():
    g = MiniPCFG()
    for lhs, entries in _TOY_CNF_RULES["binary"].items():
        for (b, c), p in entries:
            g.add_binary(lhs, b, c, p)
    for lhs, entries in _TOY_CNF_RULES["lexical"].items():
        for w, p in entries:
            g.add_lexical(lhs, w, p)
    return g


def mini_cky_parse(tokens, grammar: MiniPCFG):
    """Probabilistic CKY with unary closure. Out-of-vocabulary tokens are mapped
    to UNK. Returns (MiniTree_or_None, logprob_or_None); never raises."""
    n = len(tokens)
    if n == 0:
        return None, None
    toks = [t if t in grammar.vocab else UNK for t in tokens]
    chart = [[dict() for _ in range(n + 1)] for _ in range(n + 1)]

    def close_unary(cell):
        changed = True
        while changed:
            changed = False
            for child, (clp, _) in list(cell.items()):
                for lhs, rlp in grammar.unary_by_child.get(child, ()):
                    lp = rlp + clp
                    cur = cell.get(lhs)
                    if cur is None or lp > cur[0] + 1e-12:
                        cell[lhs] = (lp, ("UN", child))
                        changed = True

    for i in range(n):
        cell = chart[i][i + 1]
        for lhs, lp in grammar.preterminals_for(toks[i]):
            if lhs not in cell or lp > cell[lhs][0]:
                cell[lhs] = (lp, ("LEX", tokens[i]))
        close_unary(cell)

    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            cell = chart[i][j]
            for k in range(i + 1, j):
                left, right = chart[i][k], chart[k][j]
                if not left or not right:
                    continue
                for b, (blp, _) in left.items():
                    for c, (clp, _) in right.items():
                        for lhs, rlp in grammar.binary_by_children.get((b, c), ()):
                            lp = rlp + blp + clp
                            cur = cell.get(lhs)
                            if cur is None or lp > cur[0]:
                                cell[lhs] = (lp, ("BIN", b, c, k))
            close_unary(cell)

    root = chart[0][n].get(grammar.start)
    if root is None:
        return None, None

    def build(i, j, lhs):
        lp, bp = chart[i][j][lhs]
        if bp[0] == "LEX":
            return MiniTree(lhs, [bp[1]])
        if bp[0] == "UN":
            return MiniTree(lhs, [build(i, j, bp[1])])
        _, b, c, k = bp
        return MiniTree(lhs, [build(i, k, b), build(k, j, c)])

    return build(0, n, grammar.start), root[0]


# Treebank preprocessing. The raw PTB trees contain things the live stream never has:
# traces (-NONE-), punctuation, function tags, capitalised words. Leaving them in makes
# almost every sentence unparseable.
_DROP_LABELS = {",", ".", ":", "``", "''", "-LRB-", "-RRB-", "#", "$", "-NONE-"}


def _clean_tree(t):
    from nltk import Tree
    if isinstance(t, str):
        return t.lower()
    label = t.label()
    if label in _DROP_LABELS:
        return None
    if not label.startswith("-"):
        label = re.split(r"[-=]", label)[0] or label
    kids = [k for k in (_clean_tree(c) for c in t) if k is not None]
    if not kids:
        return None
    return Tree(label, kids)


def train_pcfg_from_treebank(max_sents=None):
    """Induce a PCFG from the NLTK Penn Treebank sample. Falls back to the tiny toy
    grammar if NLTK/treebank is unavailable. Returns (MiniPCFG, is_real_treebank)."""
    try:
        import nltk
        from nltk.corpus import treebank
        from nltk import Nonterminal, induce_pcfg, Tree
        try:
            treebank.parsed_sents()
        except LookupError:
            nltk.download('treebank')
        parsed = treebank.parsed_sents()
        if max_sents:
            parsed = parsed[:max_sents]
        trees = []
        for tree in parsed:
            t = _clean_tree(tree)
            if t is not None and len(t.leaves()) >= 2:
                trees.append(t)
        counts = Counter(w for t in trees for w in t.leaves())
        productions = []
        for t in trees:
            for pos in t.treepositions("leaves"):
                if counts[t[pos]] == 1:      # singletons -> <unk>, so unseen words can parse
                    t[pos] = UNK
            t = Tree("ROOT", [t])            # single start symbol for every sentence type
            t.collapse_unary(collapsePOS=False)
            t.chomsky_normal_form(horzMarkov=2)
            productions += t.productions()
        grammar = pcfg_from_nltk(induce_pcfg(Nonterminal("ROOT"), productions), "ROOT")
        print(f"[pcfg] induced grammar from {len(trees)} Penn Treebank sentences "
              f"({len(productions)} productions, {len(grammar.vocab)} lexical types)")
        return grammar, True
    except Exception as e:
        print(f"[pcfg] NLTK/treebank unavailable ({e}); using small fallback CNF grammar")
        return _build_fallback_mini_pcfg(), False


def parse_with_pcfg(tokens, grammar):
    """Most-probable parse via CKY. Returns (tree_or_None, logprob_or_None); never raises."""
    try:
        return mini_cky_parse(list(tokens), grammar)
    except Exception:
        return None, None


# Q1's coarse Universal tags and the PCFG's PTB tags are both projected into the same
# coarse space via `brown_tag_to_universal` (PTB tags share Brown's prefixes), with three
# PTB overrides that the prefix rule gets wrong. This loses PTB's fine distinctions
# (VBZ vs VBP vs VBD all become VERB), the accuracy cost of not keeping a second table.
_PTB_OVERRIDES = {"PRP": "PRON", "PRP$": "PRON", "POS": "PRT"}
_UNIVERSAL = {"NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON", "NUM", "CONJ", "PRT", "X", "."}


def reconcile_tags(q1_word_tag_pairs, pcfg_tree):
    """Compare Q1's per-word tags with the PCFG parse's preterminal tags in Universal space.
    Returns [(word, q1_tag, pcfg_tag, agree)]."""
    if pcfg_tree is None:
        return []
    leaves_with_tags = [(st.leaves()[0], st.label())
                        for st in pcfg_tree.subtrees(lambda t: t.is_preterminal())]
    out = []
    for (word, q1_tag), (_leaf, ptb_tag) in zip(q1_word_tag_pairs, leaves_with_tags):
        q1_u = q1_tag if q1_tag in _UNIVERSAL else brown_tag_to_universal(q1_tag)
        ptb_u = _PTB_OVERRIDES.get(ptb_tag) or brown_tag_to_universal(ptb_tag)
        out.append((word, q1_tag, ptb_tag, q1_u == ptb_u))
    return out


# ---- Decision rule (Part 4) --------------------------------------------

def choose_verdict(pcfg_logprob, bigram_ppl, trigram_ppl, ppl_alert_threshold):
    """1. If the PCFG parsed the sentence, prefer it (a parse is a structural signal).
       2. Else use the trigram verdict if its perplexity is not a wild outlier.
       3. Else fall back to bigram (degrades more gracefully on sparse sequences)."""
    if pcfg_logprob is not None:
        method, grammatical = "PCFG", True
    elif trigram_ppl is not None and trigram_ppl < ppl_alert_threshold * 3:
        method, grammatical = "trigram", trigram_ppl < ppl_alert_threshold
    else:
        method, grammatical = "bigram", bigram_ppl is not None and bigram_ppl < ppl_alert_threshold
    return method, ("OK" if grammatical else "FLAGGED")


# ======================================================================
# D. ORCHESTRATION
# ======================================================================

class Models:
    """Every trained model, built once and shared by all sub-systems."""

    def __init__(self):
        self.seg_lm = None
        self.hmm = None
        self.spelling = None
        self.q4_bigram_lm = None
        self.q4_trigram_lm = None
        self.pcfg_grammar = None
        self.pcfg_from_real_treebank = False
        self.avg_train_trigram_ppl = None
        self.avg_train_bigram_ppl = None


def build_models(max_word_len=12, verbose=True):
    """One-time training entry point (Q1 models, Q3 models, Q4 LMs, PCFG)."""
    m = Models()

    english_data = load_english_data()
    random.shuffle(english_data)
    split = int(len(english_data) * 0.8)
    train, test = english_data[:split], english_data[split:]
    if not test:
        test = train[-2:]

    words_sents = [[w for w, t in s] for s in train]

    m.seg_lm = TrigramLM(k=0.3)
    m.seg_lm.train(words_sents)
    m.hmm = TrigramHMM(k_trans=0.1, k_emit=0.05)
    m.hmm.train(train)

    m.spelling = SpellingModel(k_bigram=1)
    m.spelling.train(words_sents)

    m.q4_bigram_lm = BigramLM(k=ADD_K)
    m.q4_bigram_lm.train(words_sents)
    m.q4_trigram_lm = TrigramLM(k=ADD_K)
    m.q4_trigram_lm.train(words_sents)

    sample = words_sents[:200] if len(words_sents) > 200 else words_sents
    tri_ppls = [m.q4_trigram_lm.perplexity(s) for s in sample if s]
    bi_ppls = [m.q4_bigram_lm.perplexity(s) for s in sample if s]
    m.avg_train_trigram_ppl = sum(tri_ppls) / max(len(tri_ppls), 1)
    m.avg_train_bigram_ppl = sum(bi_ppls) / max(len(bi_ppls), 1)

    m.pcfg_grammar, m.pcfg_from_real_treebank = train_pcfg_from_treebank(max_sents=None)

    if verbose:
        print(f"Trained on {len(train)} sentences ({sum(len(s) for s in words_sents)} tokens).")
        print(f"Avg training trigram perplexity: {m.avg_train_trigram_ppl:.1f} "
              f"| avg bigram perplexity: {m.avg_train_bigram_ppl:.1f}")
    return m, test


class LiveEditorSession:
    """Feed tokens via `process_token`: SEGMENT/SPELL checks on every token, GRAMMAR /
    real-word check every `n_trigger` tokens."""

    def __init__(self, models: Models, max_word_len=12, n_trigger=GRAMMAR_TRIGGER_N):
        self.m = models
        self.max_word_len = max_word_len
        self.n_trigger = n_trigger
        self.window = []
        self.all_tokens = []
        self.token_events = []      # parallel to all_tokens: (merge_resolved, spell_corrected)
        self.alerts = []
        self.seg_spell_latencies = []
        self.grammar_latencies = []
        self._since_trigger = 0
        self.n_segmentation_merges_resolved = 0
        self.n_spelling_corrections = 0

    def process_token(self, raw_token):
        """Run SEGMENT-ALERT then SPELL-ALERT on one raw token; returns [(word, tag)]."""
        t0 = time.perf_counter()
        token = re.sub(r"[^a-z]", "", raw_token.lower())
        out_words, events = [], []

        if not token:
            self.seg_spell_latencies.append(time.perf_counter() - t0)
            return []

        # --- SEGMENT-ALERT ---
        in_vocab = token in self.m.spelling.vocab
        did_split = False
        if not in_vocab or len(token) >= 10:
            split_words, split_score = viterbi_segment(
                token, self.m.seg_lm, self.m.spelling.vocab, self.max_word_len)
            one_word_score = segmentation_score_as_one_word(token, self.m.seg_lm, self.m.spelling.vocab)
            if len(split_words) > 1 and split_score > one_word_score:
                pred_tags = self.m.hmm.viterbi_tag(split_words)
                self.alerts.append({
                    "type": "SEGMENT-ALERT",
                    "token": token,
                    "message": f"'{token}' looks merged -> split into {split_words} {list(zip(split_words, pred_tags))}",
                })
                for idx, (w, tg) in enumerate(zip(split_words, pred_tags)):
                    out_words.append((w, tg))
                    events.append((idx == 0, False))
                did_split = True
                self.n_segmentation_merges_resolved += 1

        if not did_split:
            word, spell_fixed = token, False
            # --- SPELL-ALERT (only if still not in vocab) ---
            if word not in self.m.spelling.vocab:
                corrected, changed = self.m.spelling.correct_nonword(word, method='b')
                if changed:
                    self.alerts.append({
                        "type": "SPELL-ALERT",
                        "token": word,
                        "message": f"'{word}' not recognised -> suggested correction '{corrected}'",
                    })
                    self.n_spelling_corrections += 1
                    word, spell_fixed = corrected, True
            tag = self.m.hmm.viterbi_tag([word])[0] if word else None
            out_words.append((word, tag))
            events.append((False, spell_fixed))

        self.seg_spell_latencies.append(time.perf_counter() - t0)

        for (w, _), ev in zip(out_words, events):
            self.all_tokens.append(w)
            self.token_events.append(ev)
            self.window.append(w)
        self._since_trigger += len(out_words)

        if self._since_trigger >= self.n_trigger:
            self._run_grammar_check()
            self._since_trigger = 0

        return out_words

    def _run_grammar_check(self):
        t0 = time.perf_counter()
        win = self.window[-self.n_trigger:] if len(self.window) > self.n_trigger else self.window[:]
        if len(win) >= 2:
            tri_ppl = self.m.q4_trigram_lm.perplexity(win)
            bi_ppl = self.m.q4_bigram_lm.perplexity(win)
            threshold = self.m.avg_train_trigram_ppl * PERPLEXITY_ALERT_MULTIPLIER
            if tri_ppl > threshold:
                msg = (f"window {win[:6]}{'...' if len(win) > 6 else ''} "
                       f"trigram-ppl={tri_ppl:.1f} (baseline~{self.m.avg_train_trigram_ppl:.1f}) "
                       f"bigram-ppl={bi_ppl:.1f}")
                self.alerts.append({"type": "GRAMMAR-ALERT", "token": None,
                                    "message": "implausible window: " + msg})

            # real-word check across the window (Q3-style)
            for i in range(1, len(win) - 1):
                prev_w, w, next_w = win[i - 1], win[i], win[i + 1]
                corrected, changed = self.m.spelling.correct_realword(prev_w, w, next_w, method='both')
                if changed:
                    self.alerts.append({
                        "type": "GRAMMAR-ALERT",
                        "token": w,
                        "message": f"possible real-word error: '{w}' -> '{corrected}' given context "
                                   f"'{prev_w} {w} {next_w}'",
                    })
        self.grammar_latencies.append(time.perf_counter() - t0)

    def latency_report(self):
        def avg(xs):
            return (sum(xs) / len(xs) * 1000) if xs else 0.0
        return {
            "avg_seg_spell_ms": avg(self.seg_spell_latencies),
            "avg_grammar_trigger_ms": avg(self.grammar_latencies),
            "n_tokens": len(self.seg_spell_latencies),
            "n_grammar_triggers": len(self.grammar_latencies),
        }


# ======================================================================
# End-of-passage analysis (Part 4)
# ======================================================================

def split_into_sentences(tokens, sentence_len_range=(6, 12), rng=None):
    """The simulated stream has no punctuation, so chunk it into pseudo-sentences of a
    random plausible length for the end-of-passage table."""
    rng = rng or random
    sentences, i = [], 0
    while i < len(tokens):
        n = rng.randint(*sentence_len_range)
        sentences.append(tokens[i:i + n])
        i += n
    return [s for s in sentences if s]


def analyze_passage(session: LiveEditorSession, models: Models):
    """Split the corrected token stream into sentences, score each with PCFG / bigram /
    trigram, apply the decision rule, and build the per-sentence summary rows."""
    sentences = split_into_sentences(session.all_tokens)
    rows = []
    threshold = models.avg_train_trigram_ppl * PERPLEXITY_ALERT_MULTIPLIER
    offset = 0

    for sent in sentences:
        events = session.token_events[offset:offset + len(sent)]
        offset += len(sent)
        tree, pcfg_lp = parse_with_pcfg(sent, models.pcfg_grammar)
        bi_ppl = models.q4_bigram_lm.perplexity(sent)
        tri_ppl = models.q4_trigram_lm.perplexity(sent)
        method, verdict = choose_verdict(pcfg_lp, bi_ppl, tri_ppl, threshold)
        rows.append({
            "sentence": " ".join(sent),
            "pcfg_result": f"{pcfg_lp:.2f}" if pcfg_lp is not None else "unparseable",
            "bigram_ppl": round(bi_ppl, 1),
            "trigram_ppl": round(tri_ppl, 1),
            "chosen_method": method,
            "verdict": verdict,
            "segmentation_merges_resolved": sum(1 for m_, _ in events if m_),
            "spelling_corrections_applied": sum(1 for _, s_ in events if s_),
        })
    return rows


# ======================================================================
# Speed-Demon benchmark (Part 5)
# ======================================================================

def speed_demon_benchmark(models: Models, batch_size=1000):
    """Time (a) the per-token segmentation+spelling pipeline vs (b) the grammar-trigger
    check alone, on the same batch of exactly `batch_size` corrupted/merged words."""
    vocab_list = list(models.spelling.vocab)
    batch = []
    attempts = 0
    while len(batch) < batch_size and attempts < batch_size * 50:
        attempts += 1
        w = random.choice(vocab_list)
        if random.random() < 0.5:
            corrupted = apply_random_edit(w)
        else:
            corrupted = w + random.choice(vocab_list)   # simulated merge
        if corrupted:
            batch.append(corrupted)

    sess = LiveEditorSession(models, n_trigger=10 ** 9)  # grammar trigger effectively off
    t0 = time.perf_counter()
    for tok in batch:
        sess.process_token(tok)
    time_seg_spell = time.perf_counter() - t0

    windows = [batch[i:i + GRAMMAR_TRIGGER_N] for i in range(0, len(batch), GRAMMAR_TRIGGER_N)]
    sess2 = LiveEditorSession(models)
    t0 = time.perf_counter()
    for win in windows:
        sess2.window = win
        sess2._run_grammar_check()
    time_grammar_only = time.perf_counter() - t0

    return {
        "batch_size": len(batch),
        "total_seg_spell_s": time_seg_spell,
        "avg_seg_spell_ms": time_seg_spell / len(batch) * 1000,
        "total_grammar_only_s": time_grammar_only,
        "n_grammar_windows": len(windows),
        "avg_grammar_window_ms": (time_grammar_only / len(windows) * 1000) if windows else 0.0,
    }
