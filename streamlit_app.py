"""
streamlit_app.py
=================
Live deployment of the Q4 integrated background editor.

Imports `nlp_pipeline.py` directly -- the exact same module used by
`Group_Assignment_Q4.ipynb` -- so this app is provably running the same
trained Q1/Q3 models and the same alert logic demonstrated in the notebook,
not a separate re-implementation.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import random
import time

import pandas as pd
import streamlit as st

import nlp_pipeline as P

st.set_page_config(page_title="NLP Q4 — Live Background Editor", layout="wide")


# ----------------------------------------------------------------------
# Model loading (cached -- trained once per deployment, not per interaction)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Training Q1 segmentation/POS models, Q3 spelling models, "
                                 "Q4 shared LM and PCFG (only happens once)...")
def get_models():
    return P.build_models(max_word_len=12)


models, _held_out = get_models()

if "session" not in st.session_state:
    st.session_state.session = P.LiveEditorSession(models)
if "typed_so_far" not in st.session_state:
    st.session_state.typed_so_far = ""
if "n_alerts_shown" not in st.session_state:
    st.session_state.n_alerts_shown = 0
if "sim_tokens" not in st.session_state:
    st.session_state.sim_tokens = []
if "sim_idx" not in st.session_state:
    st.session_state.sim_idx = 0


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("Settings")
st.sidebar.markdown(
    f"""
    **Merge probability (p):** `{P.MERGE_PROB}`
    **Grammar trigger interval (N):** `{P.GRAMMAR_TRIGGER_N}` tokens
    **Add-k smoothing:** `{P.ADD_K}`
    **PCFG source:** {'Penn Treebank (real)' if models.pcfg_from_real_treebank else 'offline fallback grammar'}

    (See `nlp_pipeline.py` docstring for the justification of each constant.)
    """
)

if st.sidebar.button("Reset session"):
    st.session_state.session = P.LiveEditorSession(models)
    st.session_state.typed_so_far = ""
    st.session_state.n_alerts_shown = 0
    st.session_state.sim_tokens = []
    st.session_state.sim_idx = 0
    st.rerun()

st.title("Integrated Background Editor")
st.caption(
    "Live segmentation (Q1) + spelling correction (Q3) + constituency-based grammar checking (Q4), "
    "all running on the same trained models used in the companion notebook."
)

tab_live, tab_sim, tab_analysis = st.tabs(
    ["✍️ Live typing", "🔁 Simulated typing", "📊 Final passage analysis"]
)


def render_new_alerts():
    alerts = st.session_state.session.alerts
    new = alerts[st.session_state.n_alerts_shown:]
    for a in new:
        icon = {"SEGMENT-ALERT": "🔀", "SPELL-ALERT": "✏️", "GRAMMAR-ALERT": "📐"}.get(a["type"], "⚠️")
        st.write(f"{icon} **[{a['type']}]** {a['message']}")
    st.session_state.n_alerts_shown = len(alerts)


# ----------------------------------------------------------------------
# Tab 1: live typing -- incremental processing of real user input
# ----------------------------------------------------------------------
with tab_live:
    st.markdown(
        "Type into the box below. New whitespace-delimited tokens are processed **incrementally** as "
        "you type (not only once you submit the whole passage) -- exactly as required."
    )
    text = st.text_area("Type your passage here:", height=120, key="live_text_input")

    if text != st.session_state.typed_so_far:
        already_processed = st.session_state.typed_so_far.split()
        now_tokens = text.split()
        # only process tokens that are new AND no longer being actively edited (i.e. not the last,
        # possibly-still-being-typed token), so we don't fire alerts on a half-typed word.
        stable_new_tokens = now_tokens[len(already_processed):-1] if len(now_tokens) > len(already_processed) else []
        for tok in stable_new_tokens:
            st.session_state.session.process_token(tok)
        st.session_state.typed_so_far = " ".join(now_tokens[:-1]) if now_tokens else ""

    st.subheader("Live alerts")
    render_new_alerts()

    rep = st.session_state.session.latency_report()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens processed", rep["n_tokens"])
    c2.metric("Avg seg+spell latency", f"{rep['avg_seg_spell_ms']:.3f} ms")
    c3.metric("Avg grammar-trigger latency", f"{rep['avg_grammar_trigger_ms']:.3f} ms")


# ----------------------------------------------------------------------
# Tab 2: simulated typing -- auto-play a sampled/pasted passage with merges
# ----------------------------------------------------------------------
with tab_sim:
    st.markdown(
        "Paste a passage (or sample one) and watch it 'type itself', word by word, with the "
        f"fast-typing merge simulator (`p={P.MERGE_PROB}`) occasionally dropping a space between words."
    )
    colA, colB = st.columns([3, 1])
    with colA:
        passage_text = st.text_area(
            "Passage to simulate-type:",
            value=("the quick brown fox jumps over the lazy dog and then runs away quickly into the "
                   "dark forest she eats a green salad with her friends every single day without fail"),
            height=100,
        )
    with colB:
        delay = st.slider("Delay per token (s)", 0.0, 0.5, 0.05, 0.05)
        if st.button("▶ Start / restart simulation"):
            words = passage_text.split()
            st.session_state.sim_tokens = P.simulate_fast_typing_merges(
                words, p=P.MERGE_PROB, rng=random.Random()
            )
            st.session_state.sim_idx = 0
            st.session_state.session = P.LiveEditorSession(models)
            st.session_state.n_alerts_shown = 0

    if st.session_state.sim_tokens:
        placeholder_text = st.empty()
        placeholder_alerts = st.container()
        typed_display = []
        for tok in st.session_state.sim_tokens[st.session_state.sim_idx:]:
            st.session_state.session.process_token(tok)
            typed_display.append(tok)
            st.session_state.sim_idx += 1
            placeholder_text.write("**Typed so far:** " + " ".join(typed_display))
            with placeholder_alerts:
                render_new_alerts()
            if delay:
                time.sleep(delay)


# ----------------------------------------------------------------------
# Tab 3: final analysis -- Part 4 per-sentence table
# ----------------------------------------------------------------------
with tab_analysis:
    st.markdown("Once you're done typing (either tab), click below for the end-of-passage analysis.")
    if st.button("Run final PCFG / n-gram sentence analysis"):
        session = st.session_state.session
        if not session.all_tokens:
            st.warning("No tokens processed yet -- type something in the Live typing or Simulated typing tab first.")
        else:
            rows = P.analyze_passage(session, models)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            rep = session.latency_report()
            st.write(
                f"**Segmentation merges resolved:** {session.n_segmentation_merges_resolved}  |  "
                f"**Spelling corrections applied:** {session.n_spelling_corrections}  |  "
                f"**Avg seg+spell latency:** {rep['avg_seg_spell_ms']:.3f} ms  |  "
                f"**Avg grammar-trigger latency:** {rep['avg_grammar_trigger_ms']:.3f} ms"
            )

    st.divider()
    st.subheader("Speed-Demon benchmark")
    if st.button("Run Speed-Demon benchmark (1,000-token batch)"):
        with st.spinner("Running benchmark..."):
            bench = P.speed_demon_benchmark(models, batch_size=1000)
        st.json(bench)
