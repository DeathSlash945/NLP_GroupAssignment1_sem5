"""
streamlit_app.py
=================
Live deployment of the Q4 integrated background editor.

Imports `nlp_pipeline.py` directly (the same module used by Group_Assignment_Q4.ipynb),
so the app runs the same trained models and alert logic as the notebook.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Live (per-keystroke) typing needs the `streamlit-keyup` package. Without it the app falls
back to st.text_area, which only reruns on blur / Ctrl+Enter.
"""

import random
import time

import pandas as pd
import streamlit as st

import nlp_pipeline as P

try:
    from streamlit_keyup import st_keyup
    HAS_KEYUP = True
except ImportError:
    HAS_KEYUP = False

st.set_page_config(page_title="NLP Live Background Editor", layout="wide")


@st.cache_resource(show_spinner="Training Q1 segmentation/POS models, Q3 spelling models, "
                                 "Q4 shared LM and PCFG (only happens once)...")
def get_models():
    return P.load_or_build_models(max_word_len=12)


models, _held_out = get_models()
ss = st.session_state


def new_session():
    return P.LiveEditorSession(models)


DEFAULT_PASSAGE = ("The quick brown fox jumps over the lazy dog and then runs away quickly into the "
                   "dark forest. She eats a green salad with her friends every single day.")

if "live_session" not in ss:
    ss.live_session = new_session()
    ss.live_tokens = []
    ss.live_key_n = 0
    ss.sim_session = new_session()
    ss.sim_tokens = []
    ss.sim_idx = 0
    ss.sim_passage = DEFAULT_PASSAGE
    ss.sim_source = ""
    ss.analysis = None
    ss.bench = None


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
    **Live typing:** {'per keystroke (streamlit-keyup)' if HAS_KEYUP else 'commit on blur (install streamlit-keyup)'}
    """
)

if st.sidebar.button("Reset session"):
    ss.live_session = new_session()
    ss.live_tokens = []
    ss.live_key_n += 1            # new widget key clears the typing box
    ss.sim_session = new_session()
    ss.sim_tokens = []
    ss.sim_idx = 0
    ss.analysis = None
    ss.bench = None
    st.rerun()


def render_alerts(session):
    """Show ALL alerts for the session, newest first (not just the ones new this rerun,
    otherwise every alert vanishes on the next keystroke)."""
    if not session.alerts:
        st.caption("No alerts yet.")
        return
    for a in reversed(session.alerts):
        st.write(f"**[{a['type']}]** {a['message']}")


def show_metrics(session):
    rep = session.latency_report()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens processed", rep["n_tokens"])
    c2.metric("Avg seg+spell latency", f"{rep['avg_seg_spell_ms']:.3f} ms")
    c3.metric("Avg grammar-trigger latency", f"{rep['avg_grammar_trigger_ms']:.3f} ms")


st.title("Integrated Background Editor")
st.caption(
    "Live segmentation (Q1) + spelling correction (Q3) + constituency-based grammar checking (Q4), "
    "all running on the same trained models used in the companion notebook."
)

tab_live, tab_sim, tab_analysis = st.tabs(
    ["Live typing", "Simulated typing", "Final passage analysis"]
)

# ----------------------------------------------------------------------
# Tab 1: live typing, incremental processing of real user input
# ----------------------------------------------------------------------
with tab_live:
    key = f"live_text_{ss.live_key_n}"
    if HAS_KEYUP:
        st.markdown(
            "Type below. Each word is checked as soon as you press space (end a sentence with "
            "`.` `!` or `?`). No Ctrl+Enter needed. Grammar checks run every "
            f"{P.GRAMMAR_TRIGGER_N} words."
        )
        text = st_keyup("Type your passage here:", key=key, debounce=150) or ""
    else:
        st.warning(
            "streamlit-keyup is not installed in this environment, so input only commits on blur "
            "or Ctrl+Enter. Fix: `pip install streamlit-keyup`, then restart `streamlit run`."
        )
        text = st.text_area("Type your passage here:", height=120, key=key)

    tokens = text.split()
    # the last token is only final once the text ends in whitespace
    finished = tokens if text.endswith((" ", "\n", "\t")) else tokens[:-1]
    prev = ss.live_tokens
    if finished[:len(prev)] != prev:      # user edited earlier text: re-check from scratch
        ss.live_session = new_session()
        prev = []
    for tok in finished[len(prev):]:
        ss.live_session.process_token(tok)
    ss.live_tokens = finished

    st.subheader("Live alerts")
    with st.container(height=320):
        render_alerts(ss.live_session)
    show_metrics(ss.live_session)

# ----------------------------------------------------------------------
# Tab 2: simulated typing, random passage with merged tokens
# ----------------------------------------------------------------------


def sample_passage():
    label, text = P.sample_random_passage()
    ss.sim_passage = text
    ss.sim_source = label


with tab_sim:
    st.markdown(
        "The passage 'types itself' word by word, with the fast-typing merge simulator "
        f"(`p={P.MERGE_PROB}`) occasionally dropping a space between words."
    )
    colA, colB = st.columns([3, 1])
    with colA:
        st.button("Sample random passage (Gutenberg / Brown / Reuters)", on_click=sample_passage)
        if ss.sim_source:
            st.caption(f"Source: {ss.sim_source}")
        st.text_area("Passage to simulate-type:", key="sim_passage", height=150)
    with colB:
        delay = st.slider("Delay per token (s)", 0.0, 0.5, 0.05, 0.05)
        if st.button("Start / restart simulation"):
            ss.sim_tokens = P.simulate_fast_typing_merges(
                ss.sim_passage.split(), p=P.MERGE_PROB, rng=random.Random())
            ss.sim_idx = 0
            ss.sim_session = new_session()
            ss.analysis = None

    if ss.sim_tokens:
        typed_ph, alerts_ph, metrics_ph = st.empty(), st.empty(), st.empty()

        def draw():
            typed_ph.write("**Typed so far:** " + " ".join(ss.sim_tokens[:ss.sim_idx]))
            with alerts_ph.container(height=320):
                render_alerts(ss.sim_session)
            with metrics_ph.container():
                show_metrics(ss.sim_session)

        draw()
        while ss.sim_idx < len(ss.sim_tokens):
            ss.sim_session.process_token(ss.sim_tokens[ss.sim_idx])
            ss.sim_idx += 1
            draw()
            if delay:
                time.sleep(delay)

# ----------------------------------------------------------------------
# Tab 3: final analysis, per-sentence table
# ----------------------------------------------------------------------
with tab_analysis:
    source = st.radio("Analyse:", ["Live typing", "Simulated typing"], horizontal=True)
    session = ss.live_session if source == "Live typing" else ss.sim_session

    if st.button("Run final PCFG / n-gram sentence analysis"):
        if not session.all_tokens:
            st.warning("No tokens processed yet for this source.")
            ss.analysis = None
        else:
            rep = session.latency_report()
            ss.analysis = {
                "rows": P.analyze_passage(session, models),
                "summary": (
                    f"**Segmentation merges resolved:** {session.n_segmentation_merges_resolved}  |  "
                    f"**Spelling corrections applied:** {session.n_spelling_corrections}  |  "
                    f"**Avg seg+spell latency:** {rep['avg_seg_spell_ms']:.3f} ms  |  "
                    f"**Avg grammar-trigger latency:** {rep['avg_grammar_trigger_ms']:.3f} ms"),
            }
    if ss.analysis:
        st.dataframe(pd.DataFrame(ss.analysis["rows"]))
        st.write(ss.analysis["summary"])

    st.divider()
    st.subheader("Speed-Demon benchmark")
    if st.button("Run Speed-Demon benchmark (1,000-token batch)"):
        with st.spinner("Running benchmark..."):
            ss.bench = P.speed_demon_benchmark(models, batch_size=1000)
    if ss.bench:
        st.json(ss.bench)
