"""
nlp_pipeline.py
================
Shared backend for Question 4 (Integrated Background Editor).

This module is imported by BOTH the demo notebook (Group_Assignment_Q4.ipynb)
and the Streamlit deployment (streamlit_app.py) so that the exact same trained
models / functions power the notebook analysis and the live web app -- nothing
is duplicated or re-implemented between the two.

It is organised in four blocks, mirroring the assignment:

  A. Q1 REUSED   -- trigram word-LM + Viterbi segmentation, trigram HMM POS
                     tagger (English), unchanged in spirit from Q1's notebook.
  B. Q3 REUSED   -- vocabulary/unigram/bigram models, Method A (edit-distance-1)
                     and Method B (symmetric delete) candidate generation,
                     non-word / real-word correction.
  C. Q4 NEW      -- merged-token typing simulator, live SEGMENT/SPELL/GRAMMAR
                     alerts, shared add-k smoothed bigram+trigram LM, PCFG
                     constituency parser + tagset reconciliation, per-sentence
                     decision rule, Speed-Demon benchmark.
  D. ORCHESTRATION -- `LiveEditorSession`, a small stateful class that both the
                     notebook and the Streamlit app drive token-by-token.

Design choices (documented here so both notebook & app can just import the
constants rather than re-justifying them):

  MERGE_PROB (p) = 0.08
      A typist fails to hit the spacebar in time on roughly 1 in every 12
      word-boundaries -- frequent enough that segmentation has real work to do
      on most sentences (avg sentence ~20 words -> ~1.6 merges/sentence),
      without merging so often that every alert becomes a segmentation alert
      and the grammar layer never gets exercised.

  GRAMMAR_TRIGGER_N = 15
      Re-scoring the trailing window with bigram+trigram perplexity, plus the
      real-word/edit-distance sweep, costs more than the O(1)-ish per-token
      vocab lookup used for SEGMENT/SPELL checks. Triggering every 15 tokens
      (roughly every half-to-one sentence) keeps that heavier check off the
      critical per-token path while still catching implausible windows before
      the passage ends. Smaller N catches errors sooner but raises false-alert
      churn (bigram/trigram perplexity is noisy over short windows); larger N
      delays detection. 15 was chosen empirically as the point where the
      Speed-Demon benchmark below shows grammar-check latency staying well
      under the simulated per-word typing delay.

  ADD_K (smoothing) = 0.5
      Matches the smoothing constant already used for the Q1 segmentation
      trigram LM, so perplexities from the Q4 shared LM and the Q1 LM are on
      a comparable scale.
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
PERPLEXITY_ALERT_MULTIPLIER = 2.5   # window flagged if its perplexity is this many
                                     # times the LM's own average training perplexity
REALWORD_THRESHOLD = 2.0            # matches Q3's correct_realword threshold


# ======================================================================
# A. Q1 REUSED -- trigram word LM, Viterbi segmentation, trigram HMM tagger
#    (identical logic to Q1's notebook; copied here so this module is
#    self-contained and importable without re-executing the Q1 notebook).
# ======================================================================

UNK_LOGPROB_PER_CHAR = -4.0


def unk_penalty(word):
    return UNK_LOGPROB_PER_CHAR * len(word)


class TrigramLM:
    """Trigram model over a token stream (words), add-k smoothed with
    back-off to bigram/unigram estimates for unseen contexts.
    (Reused unchanged from Q1 -- this is the same class used there both for
    segmentation scoring and, in Q4, as the shared LM.)"""

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
        lp = self.sentence_logprob(words)
        return math.exp(-lp / n)


class BigramLM:
    """Simple add-k smoothed bigram LM (kept as a distinct, smaller class from
    TrigramLM so Q4's 'one bigram model + one trigram model' requirement is
    unambiguous about which is which; TrigramLM.logprob(w, u) alone would also
    give bigram probabilities via back-off, but this class is the one Q4
    actually trains and reports as 'the bigram model')."""

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
        lp = self.sentence_logprob(words)
        return math.exp(-lp / n)


def viterbi_segment(chars, lm, vocab, max_word_len=15):
    """Second-order (trigram) Viterbi word segmentation. Identical algorithm
    to Q1's segmenter -- this *is* Q1's trained decoder, just imported here."""
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
        return list(chars)

    best_word = max(dp[n], key=lambda w: dp[n][w][0])
    words = []
    j, w = n, best_word
    while j > 0:
        sc, u, i = dp[j][w]
        words.append(chars[i:j])
        j, w = i, u
    words.reverse()
    return words, dp[n][best_word][0]  # (words, best total score) for the "better than one word" check


def segmentation_score_as_one_word(token, lm, vocab):
    """Score of treating `token` as a single (possibly OOV) word, for
    comparison against the best multi-word split -- used by the
    SEGMENT-ALERT trigger."""
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
    """Self-contained Brown/PTB -> Universal (coarse) tag mapping. Reused
    unchanged from Q1. Penn Treebank tags (NN, VB, JJ, IN, DT, ...) share the
    same prefixes as Brown tags, so this same function doubles as the Q1<->PTB
    tagset-reconciliation mapper used in Part 2 of Q4 below."""
    t = tag.split('-')[0].split('+')[0].rstrip('*').rstrip('$')
    if not t or not t[0].isalpha():
        return "."
    for prefix, universal in _BROWN_TO_UNIVERSAL_PREFIXES:
        if t.startswith(prefix):
            return universal
    return "X"


class TrigramHMM:
    """Trigram HMM POS tagger with beam-pruned Viterbi decoding. Identical to
    Q1's tagger -- this is Q1's trained decoder, imported here."""

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
    """Load Brown corpus tagged sentences (Universal coarse tags), falling
    back to a small embedded corpus if NLTK / the Brown download isn't
    available offline. Identical loader to Q1."""
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
# B. Q3 REUSED -- vocabulary/unigram/bigram, Method A / Method B candidate
#    generation, non-word and real-word correction (identical to Q3).
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
    """Bundles Q3's vocabulary, unigram/bigram counts, and both candidate
    generation methods, trained once and reused by every sub-system in Q4
    (SEGMENT-ALERT's OOV fallback, SPELL-ALERT, and the real-word check
    inside GRAMMAR-ALERT)."""

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
        # Method B preprocessing (SymSpell): one-time deletes dictionary
        for w in self.vocab:
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
        # Q4 note: Method B (symmetric delete) is used live by default -- it is
        # ~O(word_len) at query time vs Method A's O(word_len * 26), and the
        # Speed-Demon benchmark (Part 4) shows this matters once checks run on
        # every single typed token rather than a offline batch. Method A
        # remains available (method='a' or 'both') for the accuracy comparison.
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

        original_score = phrase_score(word)
        best_word, best_score = word, original_score
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
# C. Q4 NEW -- merged-token typing simulator, live alerts, PCFG parser,
#    decision rule, Speed-Demon benchmark.
# ======================================================================

def simulate_fast_typing_merges(words, p=MERGE_PROB, rng=None):
    """Between any two consecutive words, drop the space with probability p,
    mimicking a typist who occasionally doesn't hit the spacebar in time.
    Returns the resulting token stream (merged tokens are longer strings)."""
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
#
# Two interchangeable backends, both exposing the same minimal Tree API
# (`.label()`, `.leaves()`, `.subtrees(filter)`, `.height()`) so every
# downstream function (reconcile_tags, analyze_passage, ...) works
# unchanged regardless of which backend trained/parsed a given sentence:
#
#   * REAL backend: nltk.induce_pcfg on nltk.corpus.treebank (CNF via
#     tree.chomsky_normal_form()) + nltk.ViterbiParser. Used whenever NLTK
#     and the Penn Treebank sample are available (the normal case once this
#     runs with internet / after `nltk.download('treebank')`).
#   * FALLBACK backend: a small self-contained MiniPCFG (already in CNF) +
#     a from-scratch probabilistic CKY parser (`mini_cky_parse`), so the
#     notebook still runs end-to-end offline for demonstration when NLTK
#     isn't available -- exactly the same reasoning Q1's notebook used for
#     its own corpus fallbacks.

class MiniTree:
    """Minimal nltk.Tree-alike: a labelled node with either a single string
    leaf child (preterminal, e.g. NN -> 'fox') or exactly two MiniTree
    children (binary, CNF-style)."""

    __slots__ = ("_label", "children")

    def __init__(self, label, children):
        self._label = label
        self.children = children  # list[MiniTree] or [str] for a preterminal

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


class MiniPCFG:
    """A tiny CNF PCFG: binary rules A -> B C [p], and preterminal / lexical
    rules A -> 'word' [p]. Both stored as dict[lhs] -> list[(rhs, logprob)]."""

    def __init__(self):
        self.binary = defaultdict(list)   # lhs -> [((B, C), logprob)]
        self.lexical = defaultdict(list)  # lhs -> [(word, logprob)]
        self.start = "S"

    def add_binary(self, lhs, b, c, prob):
        self.binary[lhs].append(((b, c), math.log(prob)))

    def add_lexical(self, lhs, word, prob):
        self.lexical[lhs].append((word, math.log(prob)))

    def preterminals_for(self, word):
        return [(lhs, lp) for lhs, entries in self.lexical.items()
                for w, lp in entries if w == word]


_TOY_CNF_RULES = {
    # binary rules: lhs -> [((B, C), prob), ...]
    "binary": {
        "S": [(("NP", "VP"), 1.0)],
        "NP": [(("DT", "NN"), 0.35), (("DT", "NNJ"), 0.25), (("NP", "PP"), 0.1)],
        "NNJ": [(("JJ", "NN"), 1.0)],
        "VP": [(("VBZ", "NP"), 0.25), (("VBD", "PP"), 0.15), (("VBP", "NP"), 0.2),
               (("VBZ", "PP"), 0.1), (("VBD", "NPP"), 0.15)],
        "NPP": [(("NP", "PP"), 1.0)],
        "PP": [(("IN", "NP"), 1.0)],
    },
    # unary/lexical rules: lhs -> [(word, prob), ...] (probabilities per-lhs sum to <=1)
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
    """Probabilistic CKY / Viterbi parse for a CNF MiniPCFG. Returns
    (MiniTree_or_None, logprob_or_None); never raises on failure."""
    n = len(tokens)
    if n == 0:
        return None, None
    # chart[i][j][X] = (logprob, backpointer) for the best derivation of
    # X spanning tokens[i:j]  (0 <= i < j <= n)
    chart = [[dict() for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(n):
        for lhs, lp in grammar.preterminals_for(tokens[i]):
            best = chart[i][i + 1].get(lhs)
            if best is None or lp > best[0]:
                chart[i][i + 1][lhs] = (lp, ("LEX", tokens[i]))

    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                left_cell, right_cell = chart[i][k], chart[k][j]
                if not left_cell or not right_cell:
                    continue
                for lhs, entries in grammar.binary.items():
                    for (b, c), rule_lp in entries:
                        if b in left_cell and c in right_cell:
                            lp = rule_lp + left_cell[b][0] + right_cell[c][0]
                            cur = chart[i][j].get(lhs)
                            if cur is None or lp > cur[0]:
                                chart[i][j][lhs] = (lp, ("BIN", b, c, k))

    root = chart[0][n].get(grammar.start)
    if root is None:
        return None, None

    def build(i, j, lhs):
        lp, bp = chart[i][j][lhs]
        if bp[0] == "LEX":
            return MiniTree(lhs, [bp[1]])
        _, b, c, k = bp
        return MiniTree(lhs, [build(i, k, b), build(k, j, c)])

    tree = build(0, n, grammar.start)
    return tree, root[0]


def train_pcfg_from_treebank(max_sents=None):
    """Induce a PCFG from the NLTK Penn Treebank sample (real backend). Falls
    back to a small self-contained CNF grammar + from-scratch CKY parser
    (no NLTK dependency at all) if NLTK / the treebank corpus isn't
    available offline, so the notebook still runs end-to-end without
    internet access. Returns (grammar_object, is_real_treebank: bool)."""
    try:
        import nltk
        from nltk.corpus import treebank
        from nltk import Nonterminal, induce_pcfg
        try:
            treebank.parsed_sents()
        except LookupError:
            nltk.download('treebank')
        productions = []
        parsed = treebank.parsed_sents()
        if max_sents:
            parsed = parsed[:max_sents]
        for tree in parsed:
            tree.collapse_unary(collapsePOS=False)
            tree.chomsky_normal_form(horzMarkov=2)
            productions += tree.productions()
        S = Nonterminal('S')
        grammar = induce_pcfg(S, productions)
        print(f"[pcfg] induced grammar from {len(parsed)} Penn Treebank sentences "
              f"({len(productions)} productions)")
        return grammar, True
    except Exception as e:
        print(f"[pcfg] NLTK/treebank unavailable ({e}); using small fallback CNF grammar "
              f"+ from-scratch CKY parser")
        return _build_fallback_mini_pcfg(), False


def parse_with_pcfg(tokens, grammar):
    """Most-probable-parse via bottom-up Viterbi CKY. Dispatches to NLTK's
    ViterbiParser for a real induced nltk PCFG, or to `mini_cky_parse` for
    the offline MiniPCFG fallback. Returns (tree_or_None, logprob_or_None).
    Never raises -- unparsable sentences are reported, not crashed on."""
    if isinstance(grammar, MiniPCFG):
        return mini_cky_parse(tokens, grammar)
    try:
        import nltk
        parser = nltk.ViterbiParser(grammar)
        parses = list(parser.parse(tokens))
        if not parses:
            return None, None
        tree = parses[0]
        prob = tree.prob()
        return tree, (math.log(prob) if prob > 0 else float('-inf'))
    except Exception:
        return None, None


# Q1's coarse Universal tags <-> PTB tags used by the PCFG's lexicon both
# reduce to the same coarse space via `brown_tag_to_universal` (PTB tags share
# Brown's prefixes), so reconciliation is: map both sides through that one
# function and compare in Universal-tag space. This loses PTB's fine-grained
# distinctions (e.g. VBZ vs VBP vs VBD are all just "VERB"), which is exactly
# the accuracy loss this approach trades away in return for not needing a
# second bespoke lookup table.
def reconcile_tags(q1_word_tag_pairs, pcfg_tree):
    """Compare Q1's per-word Universal tags against the POS tags the PCFG
    parse assigned to its leaves (if it parsed), both projected into the
    coarse Universal tagset. Returns a list of (word, q1_tag, pcfg_tag,
    agree: bool)."""
    if pcfg_tree is None:
        return []
    leaves_with_tags = []
    for subtree in pcfg_tree.subtrees(lambda t: t.height() == 2):
        leaves_with_tags.append((subtree.leaves()[0], subtree.label()))
    out = []
    for (word, q1_tag), (leaf_word, ptb_tag) in zip(q1_word_tag_pairs, leaves_with_tags):
        q1_u = q1_tag if q1_tag in {"NOUN", "VERB", "ADJ", "ADV", "ADP", "DET",
                                     "PRON", "NUM", "CONJ", "PRT", "X", "."} else brown_tag_to_universal(q1_tag)
        ptb_u = brown_tag_to_universal(ptb_tag)
        out.append((word, q1_tag, ptb_tag, q1_u == ptb_u))
    return out


# ---- Decision rule (Part 4) --------------------------------------------

def choose_verdict(pcfg_logprob, bigram_ppl, trigram_ppl, ppl_alert_threshold):
    """Documented decision rule:
       1. If the PCFG parsed the sentence AND its log-prob isn't an extreme
          outlier (a parse always exists for short/simple sentences and its
          log-prob is a structural signal), PREFER the PCFG verdict.
       2. Else, if the trigram model has adequate coverage (perplexity below
          the alert threshold), use the trigram verdict.
       3. Else fall back to the bigram verdict (bigram degrades more
          gracefully than trigram on sparse/unseen sequences).
    """
    if pcfg_logprob is not None:
        method = "PCFG"
        grammatical = True  # a successful parse is itself the grammaticality signal
    elif trigram_ppl is not None and trigram_ppl < ppl_alert_threshold * 3:
        method = "trigram"
        grammatical = trigram_ppl < ppl_alert_threshold
    else:
        method = "bigram"
        grammatical = bigram_ppl is not None and bigram_ppl < ppl_alert_threshold
    verdict = "OK" if grammatical else "FLAGGED"
    return method, verdict


# ======================================================================
# D. ORCHESTRATION -- LiveEditorSession drives the whole pipeline
#    token-by-token; used identically by the notebook's simulated-typing
#    demo and by streamlit_app.py's live text box.
# ======================================================================

class Models:
    """Container for every trained model, built once and passed around --
    this IS the 'reuse trained Q1/Q3 models, do not retrain' requirement made
    concrete: one object, built once, shared by every sub-system."""

    def __init__(self):
        self.seg_lm = None          # Q1 trigram LM (segmentation scoring)
        self.hmm = None             # Q1 trigram HMM POS tagger
        self.spelling = None        # Q3 SpellingModel (vocab/unigram/bigram/A/B)
        self.q4_bigram_lm = None    # Q4 shared bigram LM (grammar alerts)
        self.q4_trigram_lm = None   # Q4 shared trigram LM (grammar alerts)
        self.pcfg_grammar = None
        self.pcfg_from_real_treebank = False
        self.avg_train_trigram_ppl = None
        self.avg_train_bigram_ppl = None


def build_models(max_word_len=12, verbose=True):
    """One-time training entry point. Trains Q1's segmentation LM + POS HMM
    on Brown, Q3's spelling models on Brown, and Q4's own shared bigram +
    trigram LMs on Brown -- everything downstream re-uses these, nothing is
    retrained per-call."""
    m = Models()

    english_data = load_english_data()
    random.shuffle(english_data)
    split = int(len(english_data) * 0.8)
    train, test = english_data[:split], english_data[split:]
    if not test:
        test = train[-2:]

    words_sents = [[w for w, t in s] for s in train]

    # --- Q1 models (segmentation + POS), reused as-is ---
    m.seg_lm = TrigramLM(k=0.3)
    m.seg_lm.train(words_sents)
    m.hmm = TrigramHMM(k_trans=0.1, k_emit=0.05)
    m.hmm.train(train)

    # --- Q3 model (spelling), reused as-is ---
    m.spelling = SpellingModel(k_bigram=1)
    m.spelling.train(words_sents)

    # --- Q4's own shared bigram + trigram LM, add-k smoothed, for grammar
    #     alerting / final sentence analysis (separate from Q1's seg_lm,
    #     which stays dedicated to segmentation scoring as instructed) ---
    m.q4_bigram_lm = BigramLM(k=ADD_K)
    m.q4_bigram_lm.train(words_sents)
    m.q4_trigram_lm = TrigramLM(k=ADD_K)
    m.q4_trigram_lm.train(words_sents)

    # Average training-set perplexity, used as the "normal" baseline that
    # GRAMMAR-ALERT windows are compared against.
    sample = words_sents[:200] if len(words_sents) > 200 else words_sents
    tri_ppls = [m.q4_trigram_lm.perplexity(s) for s in sample if s]
    bi_ppls = [m.q4_bigram_lm.perplexity(s) for s in sample if s]
    m.avg_train_trigram_ppl = sum(tri_ppls) / max(len(tri_ppls), 1)
    m.avg_train_bigram_ppl = sum(bi_ppls) / max(len(bi_ppls), 1)

    # --- PCFG (Part 2) ---
    m.pcfg_grammar, m.pcfg_from_real_treebank = train_pcfg_from_treebank(max_sents=400)

    if verbose:
        print(f"Trained on {len(train)} sentences ({sum(len(s) for s in words_sents)} tokens).")
        print(f"Avg training trigram perplexity: {m.avg_train_trigram_ppl:.1f} "
              f"| avg bigram perplexity: {m.avg_train_bigram_ppl:.1f}")
    return m, test


class LiveEditorSession:
    """Stateful driver: feed tokens one at a time via `process_token`, and it
    runs the SEGMENT/SPELL checks on every token and the GRAMMAR/real-word
    check every GRAMMAR_TRIGGER_N tokens, exactly as specified in Part 1."""

    def __init__(self, models: Models, max_word_len=12, n_trigger=GRAMMAR_TRIGGER_N):
        self.m = models
        self.max_word_len = max_word_len
        self.n_trigger = n_trigger
        self.window = []            # trailing corrected-token window (for grammar/real-word check)
        self.all_tokens = []        # full corrected token stream so far
        self.alerts = []            # list of dicts: {type, token/idx, message}
        self.seg_spell_latencies = []
        self.grammar_latencies = []
        self._since_trigger = 0
        self.n_segmentation_merges_resolved = 0
        self.n_spelling_corrections = 0

    def process_token(self, raw_token):
        """Run the per-token SEGMENT-ALERT then SPELL-ALERT checks (in that
        order, as specified), returning the list of output (word, tag-or-None)
        pairs this raw token expanded into."""
        t0 = time.perf_counter()
        token = raw_token.lower()
        token = re.sub(r"[^a-z]", "", token)
        out_words = []

        if not token:
            self.seg_spell_latencies.append(time.perf_counter() - t0)
            return []

        # --- SEGMENT-ALERT ---
        in_vocab = token in self.m.spelling.vocab
        unusually_long = len(token) >= 10
        did_split = False
        if not in_vocab or unusually_long:
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
                for w, tg in zip(split_words, pred_tags):
                    out_words.append((w, tg))
                did_split = True
                self.n_segmentation_merges_resolved += 1

        if not did_split:
            word = token
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
                    word = corrected
            tag = self.m.hmm.viterbi_tag([word])[0] if word else None
            out_words.append((word, tag))

        self.seg_spell_latencies.append(time.perf_counter() - t0)

        for w, _ in out_words:
            self.all_tokens.append(w)
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
            flagged = tri_ppl > threshold
            msg = (f"window {win[:6]}{'...' if len(win) > 6 else ''} "
                   f"trigram-ppl={tri_ppl:.1f} (baseline~{self.m.avg_train_trigram_ppl:.1f}) "
                   f"bigram-ppl={bi_ppl:.1f}")
            if flagged:
                self.alerts.append({"type": "GRAMMAR-ALERT", "token": None, "message": "implausible window: " + msg})

            # --- real-word check across the window (Q3-style) ---
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
    """The live stream has no punctuation (it's simulated typing of a
    space-only passage), so we chunk the corrected token stream into
    pseudo-sentences of a random plausible length purely for the
    end-of-passage per-sentence table."""
    rng = rng or random
    sentences, i = [], 0
    while i < len(tokens):
        n = rng.randint(*sentence_len_range)
        sentences.append(tokens[i:i + n])
        i += n
    return [s for s in sentences if s]


def analyze_passage(session: LiveEditorSession, models: Models):
    """Part 4: split the final, already segmentation- and spelling-corrected
    token stream into sentences and score each with PCFG / bigram / trigram,
    apply the decision rule, and build the summary table."""
    sentences = split_into_sentences(session.all_tokens)
    rows = []
    threshold = models.avg_train_trigram_ppl * PERPLEXITY_ALERT_MULTIPLIER

    for sent in sentences:
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
        })
    return rows


# ======================================================================
# Speed-Demon benchmark (Part 5)
# ======================================================================

def speed_demon_benchmark(models: Models, batch_size=1000):
    """Builds a batch of exactly `batch_size` simulated misspelled/merged
    words, then times (a) the full per-token live-check pipeline
    (segmentation-check + spelling-check) vs (b) the grammar-trigger check
    alone, in isolation."""
    vocab_list = list(models.spelling.vocab)
    batch = []
    attempts = 0
    while len(batch) < batch_size and attempts < batch_size * 50:
        attempts += 1
        w = random.choice(vocab_list)
        if random.random() < 0.5:
            corrupted = apply_random_edit(w)
        else:
            # simulate a merge: glue two random vocab words together
            corrupted = w + random.choice(vocab_list)
        if corrupted:
            batch.append(corrupted)

    # (a) full per-token pipeline (fresh session, no grammar triggering)
    sess = LiveEditorSession(models, n_trigger=10 ** 9)  # effectively disable grammar trigger
    t0 = time.perf_counter()
    for tok in batch:
        sess.process_token(tok)
    time_seg_spell = time.perf_counter() - t0

    # (b) grammar-trigger check alone, run on windows drawn from the same batch
    window_size = GRAMMAR_TRIGGER_N
    windows = [batch[i:i + window_size] for i in range(0, len(batch), window_size)]
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
