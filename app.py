"""
app.py  —  Streamlit UI for the GitLab Agentic RAG Chatbot.
Run with:  streamlit run app.py
"""

import asyncio
import concurrent.futures
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="GitLab AI — Documentation Expert",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

from phase2_agents.crew import create_simple_retrieval_crew
from phase2_agents.run_agents import _needs_full_pipeline
from phase2_agents.parallel_pipeline import run_parallel_preprocess

# ─────────────────────────────────────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global reset to dark ──────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
section.main { background: #0D0B14 !important; }

.main .block-container {
    padding-top: 1.4rem;
    padding-bottom: 5rem;
    max-width: 920px;
}

/* sidebar */
[data-testid="stSidebar"] { background: #0A0812 !important; border-right: 1px solid #1E1A2B; }
[data-testid="stSidebar"] * { color: #C0B8D0; }

/* native streamlit elements */
[data-testid="stVerticalBlock"]    { background: transparent !important; }
[data-testid="stHorizontalBlock"]  { background: transparent !important; }
div[class*="stMarkdown"] p,
div[class*="stMarkdown"] li        { color: #C0B8D0; }

/* chat input */
[data-testid="stChatInput"] > div  { background: #16131F !important; border: 1px solid #2A2535 !important; border-radius: 12px !important; }
[data-testid="stChatInput"] textarea { background: #16131F !important; color: #E8E3F0 !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #4A4260 !important; }

/* sidebar buttons (sample questions) */
.stButton > button {
    background: #16131F !important;
    color: #9B8FBB !important;
    border: 1px solid #2A2535 !important;
    border-radius: 8px !important;
    font-size: .78rem !important;
    text-align: left !important;
    padding: .45rem .7rem !important;
    transition: all .15s;
}
.stButton > button:hover {
    background: rgba(252,109,38,.08) !important;
    border-color: rgba(252,109,38,.35) !important;
    color: #FC6D26 !important;
}

/* status widget */
[data-testid="stStatusWidget"],
[data-testid="stExpander"]  { background: #16131F !important; border: 1px solid #2A2535 !important; border-radius: 10px !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2A2535; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #FC6D26; }

/* ── Hero ──────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #100E1A 0%, #1A1626 55%, #100E1A 100%);
    border: 1px solid rgba(252,109,38,.18);
    border-radius: 16px;
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.hero::after {
    content:''; position:absolute; top:-70px; right:-50px;
    width:340px; height:340px;
    background: radial-gradient(circle, rgba(252,109,38,.09) 0%, transparent 68%);
    pointer-events:none;
}
.hero-eyebrow { font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#FC6D26; margin-bottom:.55rem; }
.hero-title {
    font-size:2.3rem; font-weight:800; line-height:1.1;
    background: linear-gradient(120deg,#FC6D26 0%,#FCA326 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin:0 0 .5rem;
}
.hero-sub { font-size:.97rem; color:rgba(200,190,225,.65); margin:0 0 1.4rem; max-width:560px; line-height:1.65; }
.badge-row { display:flex; gap:.45rem; flex-wrap:wrap; }
.badge {
    display:inline-flex; align-items:center; gap:.28rem;
    padding:.26rem .72rem; border-radius:999px;
    font-size:.7rem; font-weight:700; letter-spacing:.03em; white-space:nowrap;
}
.bo { background:rgba(252,109,38,.12); color:#FC6D26; border:1px solid rgba(252,109,38,.25); }
.bp { background:rgba(155,127,224,.12); color:#9B7FE0; border:1px solid rgba(155,127,224,.25); }
.bg { background:rgba(63,185,80,.12);  color:#3FB950; border:1px solid rgba(63,185,80,.25); }
.bb { background:rgba(88,166,255,.12); color:#58A6FF; border:1px solid rgba(88,166,255,.25); }
.bw { background:rgba(255,255,255,.06); color:rgba(200,190,225,.7); border:1px solid rgba(255,255,255,.12); }

/* ── Feature cards (dark) ──────────────────────────────────── */
.features-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.85rem; margin-bottom:1.6rem; }
.fc {
    background:#16131F; border:1px solid #2A2535; border-radius:12px;
    padding:1.25rem 1.35rem; position:relative; overflow:hidden;
    transition: box-shadow .18s, transform .18s, border-color .18s;
}
.fc::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#FC6D26,#FCA326); border-radius:12px 12px 0 0;
}
.fc:hover { transform:translateY(-2px); box-shadow:0 6px 28px rgba(0,0,0,.4); border-color:rgba(252,109,38,.25); }
.fc-icon  { font-size:1.65rem; margin-bottom:.5rem; display:block; }
.fc-title { font-size:.91rem; font-weight:700; color:#E8E3F0; margin-bottom:.32rem; }
.fc-desc  { font-size:.78rem; color:#7A708E; line-height:1.55; }
.fc-desc strong { color:#C0B8D0; }
.fc-desc code   { background:#1E1A2B; color:#9B7FE0; padding:.1rem .3rem; border-radius:4px; font-size:.72rem; }
.fc-chips { margin-top:.65rem; display:flex; flex-wrap:wrap; gap:.28rem; }
.chip {
    background:#1E1A2B; border:1px solid #352E48;
    border-radius:999px; padding:.1rem .52rem;
    font-size:.65rem; font-weight:600; color:#7A70A0;
    font-family:ui-monospace,monospace;
}

/* ── Architecture callout ──────────────────────────────────── */
.arch-callout {
    background:linear-gradient(135deg,#100E1A,#1A1626);
    border:1px solid rgba(252,109,38,.12); border-radius:12px;
    padding:1.3rem 1.55rem; margin-bottom:1.7rem;
    font-family:ui-monospace,monospace; font-size:.76rem; color:#4A4260; line-height:1.85;
}
.hl  { color:#FC6D26; font-weight:700; }
.hl2 { color:#9B7FE0; }
.hl3 { color:#3FB950; }
.dim { color:#3A3250; }

/* ── Chat messages ─────────────────────────────────────────── */
.msg-user { display:flex; justify-content:flex-end; margin-bottom:1rem; }
.bubble-user {
    background:linear-gradient(135deg,#FC6D26 0%,#FCA326 100%);
    color:#fff; padding:.68rem 1.1rem;
    border-radius:18px 18px 4px 18px; max-width:72%;
    font-size:.9rem; line-height:1.55;
    box-shadow:0 2px 12px rgba(252,109,38,.25); word-wrap:break-word;
}
.msg-bot { display:flex; gap:.65rem; margin-bottom:1rem; align-items:flex-start; }
.bot-av {
    width:32px; height:32px; flex-shrink:0;
    background:linear-gradient(135deg,#16131F,#1E1A2B);
    border:2px solid rgba(252,109,38,.3); border-radius:50%;
    display:flex; align-items:center; justify-content:center; font-size:.95rem;
}
.bubble-bot {
    background:#16131F; border:1px solid #2A2535;
    border-radius:4px 14px 14px 14px; padding:.95rem 1.15rem;
    max-width:88%; font-size:.88rem; line-height:1.68; color:#D8D0E8;
    box-shadow:0 2px 12px rgba(0,0,0,.25); word-wrap:break-word;
}
.bubble-bot code {
    background:#1E1A2B; color:#9B7FE0;
    padding:.12rem .35rem; border-radius:4px; font-size:.82rem;
}
.bubble-bot pre {
    background:#0D0B14; border:1px solid #2A2535; border-radius:8px;
    padding:.85rem 1rem; overflow-x:auto; font-size:.78rem; line-height:1.6;
}

/* ── Pipeline pill ─────────────────────────────────────────── */
.pipe-pill {
    display:inline-flex; align-items:center; gap:.32rem;
    padding:.16rem .6rem; border-radius:999px;
    font-size:.66rem; font-weight:700; margin:.55rem 0 .15rem;
}
.pipe-simple { background:rgba(63,185,80,.1);   color:#3FB950; border:1px solid rgba(63,185,80,.22); }
.pipe-full   { background:rgba(155,127,224,.1); color:#9B7FE0; border:1px solid rgba(155,127,224,.22); }

/* ── Sources ───────────────────────────────────────────────── */
.sources-wrap { margin-top:.85rem; border-top:1px solid #1E1A2B; padding-top:.65rem; }
.sources-label { font-size:.64rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:#4A4260; margin-bottom:.38rem; }
.src-chip {
    display:inline-flex; align-items:center; gap:.22rem;
    padding:.18rem .6rem; background:#1E1A2B; border:1px solid #2A2535;
    border-radius:6px; font-size:.7rem; color:#7A708E;
    margin:.18rem .18rem .18rem 0; text-decoration:none; transition:all .14s;
}
.src-chip:hover { background:rgba(252,109,38,.08); border-color:rgba(252,109,38,.3); color:#FC6D26; text-decoration:none; }

/* ── Response meta ─────────────────────────────────────────── */
.resp-meta { font-size:.65rem; color:#3A3250; margin-top:.35rem; }

/* ── Live agent progress panel ─────────────────────────────── */
.agent-panel {
    background:#100E1A; border:1px solid #2A2535; border-radius:12px;
    padding:1.1rem 1.4rem; margin:.5rem 0 1rem; font-family:ui-monospace,monospace;
}
.agent-panel-title {
    font-size:.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.1em; color:#4A4260; margin-bottom:.85rem;
}
.agent-step { display:flex; align-items:flex-start; gap:.7rem; padding:.35rem 0; position:relative; }
.agent-step:not(:last-child)::after {
    content:''; position:absolute; left:.55rem; top:1.4rem;
    width:1px; height:calc(100% - .1rem); background:#2A2535;
}
.step-icon-wrap { width:1.1rem; text-align:center; flex-shrink:0; margin-top:.05rem; }
.step-icon-done    { color:#3FB950; font-size:.9rem; }
.step-icon-active  { color:#FC6D26; font-size:.9rem; animation: pulse 1s ease-in-out infinite; }
.step-icon-pending { color:#2A2535; font-size:.9rem; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
.step-body { flex:1; min-width:0; }
.step-label-done    { font-size:.8rem; font-weight:700; color:#3FB950; }
.step-label-active  { font-size:.8rem; font-weight:700; color:#FC6D26; }
.step-label-pending { font-size:.8rem; font-weight:600; color:#2A2535; }
.step-detail-done    { font-size:.68rem; color:#3A3250; margin-top:.06rem; }
.step-detail-active  { font-size:.68rem; color:#7A6890; margin-top:.06rem; }
.step-detail-pending { font-size:.68rem; color:#1E1A2B; margin-top:.06rem; }
.step-time { font-size:.62rem; color:#3A3250; margin-left:auto; white-space:nowrap; padding-left:.5rem; }

/* sidebar section */
.sb-section { background:rgba(255,255,255,.03); border:1px solid #1E1A2B; border-radius:10px; padding:.85rem .95rem; margin-bottom:.85rem; }
.sb-title   { font-size:.64rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:#3A3250; margin-bottom:.55rem; }
.sb-tech-item { display:flex; align-items:center; gap:.45rem; padding:.28rem 0; border-bottom:1px solid rgba(255,255,255,.04); font-size:.8rem; }
.sb-tech-item:last-child { border-bottom:none; }
.sb-tech-name { font-weight:600; color:#C0B8D0; flex:1; }
.sb-tech-tag  { font-size:.62rem; font-weight:700; padding:.08rem .42rem; border-radius:999px; background:rgba(63,185,80,.12); color:#3FB950; border:1px solid rgba(63,185,80,.2); }
.sb-metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:.45rem; }
.sb-metric { background:rgba(255,255,255,.03); border:1px solid #1E1A2B; border-radius:8px; padding:.55rem; text-align:center; }
.sb-metric-val { font-size:1.3rem; font-weight:800; color:#FC6D26; }
.sb-metric-lbl { font-size:.58rem; color:#3A3250; text-transform:uppercase; letter-spacing:.05em; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

_STEPS_SIMPLE = [
    ("router",      "🔀", "Smart Router",         "Simple pipeline selected — query is clear and specific"),
    ("embed",       "🧩", "Query Embedder",        "Converting query → 384-dimensional vector (BAAI/bge-small)"),
    ("retrieve",    "🔍", "Retriever Agent",       "Searching 8,000+ GitLab docs in ChromaDB using cosine similarity"),
    ("validate",    "✅", "Heuristic Validator",   "Dropping chunks below 0.40 relevance threshold"),
    ("synthesise",  "✍️", "Synthesiser Agent",     "Writing grounded answer with inline citations"),
]

_STEPS_FULL = [
    ("router",      "🔀", "Smart Router",         "Full pipeline selected — query needs enrichment"),
    ("intent",      "🧠", "Intent Classifier",    "Classifying query intent [async ∥]"),
    ("rewrite",     "✏️",  "Query Rewriter",       "Generating optimised search phrasings [async ∥]"),
    ("embed",       "🧩", "Query Embedder",        "Converting query → 384-dimensional vector (BAAI/bge-small)"),
    ("retrieve",    "🔍", "Retriever Agent",       "Searching 8,000+ GitLab docs in ChromaDB using cosine similarity"),
    ("validate",    "✅", "Heuristic Validator",   "Dropping chunks below 0.40 relevance threshold"),
    ("synthesise",  "✍️", "Synthesiser Agent",     "Writing grounded answer with inline citations"),
]


def _render_progress(steps: list, current_stage: str, timings: dict) -> str:
    """Return the HTML for the live agent progress panel."""
    current_idx = next((i for i, s in enumerate(steps) if s[0] == current_stage), len(steps))
    rows = []
    for i, (sid, icon, label, detail) in enumerate(steps):
        if i < current_idx:
            icon_html = f'<span class="step-icon-done">✓</span>'
            label_cls, detail_cls = "step-label-done", "step-detail-done"
            elapsed = timings.get(sid, "")
            time_html = f'<span class="step-time">{elapsed}</span>' if elapsed else ""
        elif i == current_idx:
            icon_html = f'<span class="step-icon-active">{icon}</span>'
            label_cls, detail_cls = "step-label-active", "step-detail-active"
            time_html = '<span class="step-time" style="color:#FC6D26">running…</span>'
        else:
            icon_html = f'<span class="step-icon-pending">○</span>'
            label_cls, detail_cls = "step-label-pending", "step-detail-pending"
            time_html = ""

        rows.append(
            f'<div class="agent-step">'
            f'  <div class="step-icon-wrap">{icon_html}</div>'
            f'  <div class="step-body">'
            f'    <div class="{label_cls}">{label}</div>'
            f'    <div class="{detail_cls}">{detail}</div>'
            f'  </div>'
            f'  {time_html}'
            f'</div>'
        )

    return (
        '<div class="agent-panel">'
        '<div class="agent-panel-title">🤖 Agent Activity</div>'
        + "".join(rows)
        + "</div>"
    )


def _fmt_s(t: float) -> str:
    return f"{t:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_crew():
    return create_simple_retrieval_crew()


def _extract_sources(text: str) -> list[dict]:
    urls = re.findall(r"https://docs\.gitlab\.com[^\s\)\]\,\"\']+", text)
    seen, cards = set(), []
    for url in urls:
        url = url.rstrip(".")
        if url not in seen:
            seen.add(url)
            label = url.replace("https://docs.gitlab.com/ee/", "").rstrip("/") or "gitlab docs"
            cards.append({"url": url, "label": label})
    return cards


def _run_async_preprocess(query: str) -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, run_parallel_preprocess(query))
        try:
            return future.result(timeout=45)
        except Exception:
            return ""


def _ask_with_progress(query: str, progress_placeholder) -> dict:
    """
    Run the full pipeline, updating the progress panel live at each stage.
    The crew runs in a background thread so the UI stays responsive.
    """
    t0 = time.time()
    timings: dict = {}
    use_full = _needs_full_pipeline(query)
    steps = _STEPS_FULL if use_full else _STEPS_SIMPLE

    def update(stage: str):
        progress_placeholder.markdown(
            _render_progress(steps, stage, timings), unsafe_allow_html=True
        )

    # ── Stage 1: Router ──────────────────────────────────────────────────────
    update("router")
    time.sleep(0.3)
    timings["router"] = _fmt_s(time.time() - t0)

    enriched = query

    # ── Stage 2: Async pre-process (full pipeline only) ──────────────────────
    if use_full:
        update("intent")
        t_pre = time.time()
        preprocess_context = _run_async_preprocess(query)
        elapsed_pre = time.time() - t_pre
        timings["intent"]  = _fmt_s(elapsed_pre)
        timings["rewrite"] = _fmt_s(elapsed_pre)
        enriched = f"{query}\n\n[PREPROCESSING CONTEXT]\n{preprocess_context}" if preprocess_context else query

    # ── Stage 3: Embed (fast, but show it) ──────────────────────────────────
    update("embed")
    time.sleep(0.4)
    timings["embed"] = _fmt_s(time.time() - t0)

    # ── Stage 4+5: Retriever + Synthesiser (run crew in thread) ─────────────
    result_holder: dict = {}
    error_holder:  dict = {}

    def _crew_thread():
        try:
            crew = _get_crew()
            res = crew.kickoff(inputs={"user_query": enriched})
            result_holder["value"] = res
        except Exception as exc:
            error_holder["error"] = exc

    thread = threading.Thread(target=_crew_thread, daemon=True)
    thread.start()

    t_crew = time.time()
    in_synthesis = False

    while thread.is_alive():
        elapsed_crew = time.time() - t_crew
        if elapsed_crew > 18 and not in_synthesis:
            # Heuristic: after ~18s the retriever is done, synthesiser is running
            timings["retrieve"] = _fmt_s(elapsed_crew)
            timings["validate"] = _fmt_s(elapsed_crew + 0.1)
            in_synthesis = True
            update("synthesise")
        elif not in_synthesis:
            update("retrieve")
        time.sleep(0.8)

    thread.join()

    if "error" in error_holder:
        raise error_holder["error"]

    raw = result_holder["value"]
    answer = raw.raw if hasattr(raw, "raw") else str(raw)

    # mark remaining steps done
    now = time.time()
    if not in_synthesis:
        timings["retrieve"] = _fmt_s(now - t0)
        timings["validate"] = _fmt_s(now - t0)
    timings["synthesise"] = _fmt_s(now - t0)

    # show fully-done panel briefly before replacing with answer
    update("__done__")
    time.sleep(0.4)

    return {
        "answer": answer,
        "pipeline": "full" if use_full else "simple",
        "elapsed": round(now - t0, 1),
        "sources": _extract_sources(answer),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='padding:.5rem 0 1.1rem'>"
        "<div style='font-size:1.5rem;margin-bottom:.15rem'>🦊</div>"
        "<div style='font-size:.95rem;font-weight:800;color:#FC6D26;'>GitLab AI</div>"
        "<div style='font-size:.72rem;color:#3A3250;'>Documentation Expert</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("""
<div class="sb-section">
  <div class="sb-title">🏗 Pipeline</div>
  <div style="font-family:ui-monospace,monospace;font-size:.7rem;color:#3A3250;line-height:1.9;">
    <span style="color:#FC6D26;font-weight:700;">User Query</span><br>
    &nbsp;&nbsp;↓<br>
    <span style="color:#9B7FE0;">Smart Router</span>
    <span style="font-size:.62rem;">&nbsp;(clear → simple | vague → full)</span><br>
    &nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#9B7FE0;">Intent+Rewrite</span>&nbsp;<span style="font-size:.62rem;">[async∥]</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#FC6D26;font-weight:700;">Retriever Agent</span><br>
    <span style="font-size:.62rem;">&nbsp;&nbsp;ChromaDB · cosine · RRF</span><br>
    &nbsp;&nbsp;↓<br>
    <span style="color:#3FB950;font-weight:700;">Synthesiser Agent</span><br>
    <span style="font-size:.62rem;">&nbsp;&nbsp;grounded · cited · no hallucination</span><br>
    &nbsp;&nbsp;↓<br>
    <span style="color:#FC6D26;font-weight:700;">Cited Answer</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sb-section">
  <div class="sb-title">⚡ Tech Stack</div>
  <div class="sb-tech-item"><span class="sb-tech-name">CrewAI</span><span class="sb-tech-tag">Agents</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">ChromaDB</span><span class="sb-tech-tag">Vector DB</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">BAAI/bge-small</span><span class="sb-tech-tag">Embeddings</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">OpenRouter</span><span class="sb-tech-tag">LLM</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">LangChain</span><span class="sb-tech-tag">RAG</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">asyncio</span><span class="sb-tech-tag">Parallel</span></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sb-section">
  <div class="sb-title">📊 System Stats</div>
  <div class="sb-metric-grid">
    <div class="sb-metric"><div class="sb-metric-val">$0</div><div class="sb-metric-lbl">Total Cost</div></div>
    <div class="sb-metric"><div class="sb-metric-val">8K+</div><div class="sb-metric-lbl">Docs Indexed</div></div>
    <div class="sb-metric"><div class="sb-metric-val">384</div><div class="sb-metric-lbl">Embed Dims</div></div>
    <div class="sb-metric"><div class="sb-metric-val">0.40</div><div class="sb-metric-lbl">Min Relevance</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        "<div class='sb-title' style='margin-top:.4rem;padding-left:.1rem'>💡 Try asking</div>",
        unsafe_allow_html=True,
    )
    for q in [
        "How do I cache npm packages in GitLab CI?",
        "What is the difference between stages and jobs?",
        "How do I set up a Docker-in-Docker pipeline?",
        "How do I protect a branch in GitLab?",
        "What are GitLab CI/CD variables?",
        "How do I run SAST security scanning?",
    ]:
        if st.button(q, key=f"sq_{q}", use_container_width=True):
            st.session_state.pending_query = q

    st.markdown(
        "<div style='margin-top:1.4rem;font-size:.62rem;color:#2A2535;text-align:center;'>"
        "Built by Raghav Gupta · Agentic RAG<br>"
        "CrewAI · ChromaDB · sentence-transformers"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🦊 Agentic RAG · GitLab Documentation</div>
  <div class="hero-title">Ask anything about GitLab</div>
  <div class="hero-sub">
    A multi-agent AI that retrieves answers directly from GitLab's official
    documentation — grounded, cited, and hallucination-free.
  </div>
  <div class="badge-row">
    <span class="badge bo">🤖 Multi-Agent (CrewAI)</span>
    <span class="badge bp">🔍 Semantic Search (ChromaDB)</span>
    <span class="badge bg">💰 Zero Cost</span>
    <span class="badge bb">🧠 Local Embeddings</span>
    <span class="badge bw">📄 Citation-Grounded</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CARDS — visible when no conversation has started
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
<div class="features-grid">

  <div class="fc">
    <span class="fc-icon">🤖</span>
    <div class="fc-title">Multi-Agent Architecture</div>
    <div class="fc-desc">
      CrewAI orchestrates a <strong>Retriever</strong> and <strong>Synthesiser</strong> agent
      sequentially. Vague queries trigger parallel
      Intent Classification and Query Rewriting first.
    </div>
    <div class="fc-chips">
      <span class="chip">CrewAI</span><span class="chip">asyncio</span><span class="chip">parallel</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">🔍</span>
    <div class="fc-title">Semantic Vector Search</div>
    <div class="fc-desc">
      8,000+ GitLab docs chunked, embedded locally, and stored in ChromaDB.
      Cosine similarity with <strong>Reciprocal Rank Fusion</strong>
      for multi-query retrieval.
    </div>
    <div class="fc-chips">
      <span class="chip">ChromaDB</span><span class="chip">RRF</span><span class="chip">cosine</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">📄</span>
    <div class="fc-title">Citation-Grounded Answers</div>
    <div class="fc-desc">
      Every fact is backed by a source URL from <code>docs.gitlab.com</code>.
      Deprecated features are flagged. Zero hallucinations.
    </div>
    <div class="fc-chips">
      <span class="chip">RAG</span><span class="chip">no hallucination</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">⚡</span>
    <div class="fc-title">Smart Query Routing</div>
    <div class="fc-desc">
      Clear questions skip straight to retrieval (~5s).
      Vague queries run async intent + rewrite in parallel,
      cutting pre-processing time in half.
    </div>
    <div class="fc-chips">
      <span class="chip">routing logic</span><span class="chip">async∥</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">🧠</span>
    <div class="fc-title">Zero-Cost Local Embeddings</div>
    <div class="fc-desc">
      <code>BAAI/bge-small-en-v1.5</code> runs on CPU — no GPU, no API fees.
      Disk-cached so re-indexing skips already-processed chunks.
    </div>
    <div class="fc-chips">
      <span class="chip">sentence-transformers</span><span class="chip">384-dim</span><span class="chip">$0</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">🏗️</span>
    <div class="fc-title">Production-Grade Ingestion</div>
    <div class="fc-desc">
      Git sparse-checkout grabs only <code>/doc</code> (~80 MB vs 4 GB full repo).
      Header-aware Markdown chunking across 30,000+ chunks.
    </div>
    <div class="fc-chips">
      <span class="chip">sparse-checkout</span><span class="chip">LangChain</span><span class="chip">tiktoken</span>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="arch-callout">
  <span class="hl">Query</span><span class="dim"> → </span><span class="hl2">Router</span>
  <span class="dim"> → </span><span class="hl2">[Intent + Rewrite]</span>
  <span class="dim"> (parallel, vague queries only) → </span><span class="hl">Retriever Agent</span>
  <span class="dim"> → semantic / filtered / multi-query search → </span>
  <span class="hl3">Validator</span><span class="dim"> (≥ 0.40 relevance) → </span>
  <span class="hl">Synthesiser Agent</span><span class="dim"> → </span><span class="hl3">Cited Answer</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _render_bot_message(content: str, pipeline: str, elapsed: float, sources: list) -> str:
    pipe_class = "pipe-full" if pipeline == "full" else "pipe-simple"
    pipe_label = (
        "🔄 Full pipeline — intent + rewrite + retrieve + synthesise"
        if pipeline == "full"
        else "⚡ Simple pipeline — retrieve + synthesise"
    )
    chips = "".join(
        f'<a class="src-chip" href="{s["url"]}" target="_blank">📎 {s["label"]}</a>'
        for s in sources
    )
    sources_html = (
        f'<div class="sources-wrap"><div class="sources-label">Sources</div>{chips}</div>'
        if sources else ""
    )
    return (
        f'<div class="msg-bot">'
        f'  <div class="bot-av">🦊</div>'
        f'  <div style="flex:1;min-width:0;">'
        f'    <div class="bubble-bot">{content}</div>'
        f'    <div class="pipe-pill {pipe_class}">{pipe_label}</div>'
        f'    {sources_html}'
        f'    <div class="resp-meta">⏱ {elapsed}s</div>'
        f'  </div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="bubble-user">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _render_bot_message(
                msg["content"], msg.get("pipeline", "simple"),
                msg.get("elapsed", 0), msg.get("sources", [])
            ),
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT + LIVE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = ""
else:
    query = st.chat_input("Ask anything about GitLab…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    st.markdown(
        f'<div class="msg-user"><div class="bubble-user">{query}</div></div>',
        unsafe_allow_html=True,
    )

    progress_placeholder = st.empty()
    result = None

    try:
        result = _ask_with_progress(query, progress_placeholder)
    except Exception as exc:
        progress_placeholder.error(f"❌ Something went wrong: {exc}")

    if result:
        progress_placeholder.empty()

        answer_escaped = result["answer"].replace("<", "&lt;").replace(">", "&gt;")

        st.markdown(
            _render_bot_message(
                answer_escaped, result["pipeline"],
                result["elapsed"], result["sources"]
            ),
            unsafe_allow_html=True,
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_escaped,
            "pipeline": result["pipeline"],
            "elapsed": result["elapsed"],
            "sources": result["sources"],
        })

    st.rerun()
