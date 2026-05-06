"""
app.py
──────
Streamlit UI for the GitLab Agentic RAG Chatbot.

Run with:
    streamlit run app.py
"""

import asyncio
import concurrent.futures
import re
import sys
import time
from pathlib import Path

import streamlit as st

# ── Must be the very first Streamlit call ────────────────────────────────────
st.set_page_config(
    page_title="GitLab AI — Documentation Expert",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from phase2_agents.crew import create_simple_retrieval_crew
from phase2_agents.run_agents import _needs_full_pipeline
from phase2_agents.parallel_pipeline import run_parallel_preprocess

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Reset & base ─────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"] {
    background: #F9F9FB;
}

[data-testid="stSidebar"] {
    background: #16141C !important;
    border-right: 1px solid #2A2733;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 900px;
}

/* ── Hero ─────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #16141C 0%, #1F1C2C 60%, #16141C 100%);
    border: 1px solid rgba(252,109,38,.18);
    border-radius: 16px;
    padding: 2.5rem 2.8rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(252,109,38,.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #FC6D26;
    margin-bottom: .6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(120deg, #FC6D26 0%, #FCA326 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .55rem;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,.62);
    margin: 0 0 1.4rem;
    max-width: 560px;
    line-height: 1.6;
}
.badge-row { display: flex; gap: .5rem; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; gap: .3rem;
    padding: .28rem .75rem; border-radius: 999px;
    font-size: .72rem; font-weight: 700; letter-spacing: .03em;
    white-space: nowrap;
}
.bo { background: rgba(252,109,38,.14); color:#FC6D26; border:1px solid rgba(252,109,38,.28); }
.bp { background: rgba(155,127,224,.14); color:#9B7FE0; border:1px solid rgba(155,127,224,.28); }
.bg { background: rgba(63,185,80,.14); color:#3FB950; border:1px solid rgba(63,185,80,.28); }
.bb { background: rgba(88,166,255,.14); color:#58A6FF; border:1px solid rgba(88,166,255,.28); }
.bw { background: rgba(255,255,255,.08); color:rgba(255,255,255,.7); border:1px solid rgba(255,255,255,.14); }

/* ── Feature cards ────────────────────────────────────────── */
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .9rem;
    margin-bottom: 1.8rem;
}
.fc {
    background: #fff;
    border: 1px solid #EBEBEB;
    border-radius: 12px;
    padding: 1.3rem 1.4rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow .18s, transform .18s, border-color .18s;
}
.fc::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #FC6D26, #FCA326);
    border-radius: 12px 12px 0 0;
}
.fc:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(0,0,0,.08);
    border-color: rgba(252,109,38,.22);
}
.fc-icon { font-size:1.7rem; margin-bottom:.55rem; display:block; }
.fc-title { font-size:.93rem; font-weight:700; color:#1A1726; margin-bottom:.35rem; }
.fc-desc { font-size:.8rem; color:#777; line-height:1.55; }
.fc-chips { margin-top:.7rem; display:flex; flex-wrap:wrap; gap:.3rem; }
.chip {
    background:#F4F4F4; border:1px solid #E4E4E4;
    border-radius:999px; padding:.12rem .55rem;
    font-size:.68rem; font-weight:600; color:#666;
    font-family: ui-monospace, monospace;
}

/* ── Architecture callout ─────────────────────────────────── */
.arch-callout {
    background: linear-gradient(135deg, #1A1726, #221F30);
    border: 1px solid rgba(252,109,38,.15);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.8rem;
    font-family: ui-monospace, monospace;
    font-size: .78rem;
    color: #888;
    line-height: 1.8;
}
.arch-callout .hl  { color:#FC6D26; font-weight:700; }
.arch-callout .hl2 { color:#9B7FE0; }
.arch-callout .hl3 { color:#3FB950; }
.arch-callout .dim { color:#555; }

/* ── Chat messages ────────────────────────────────────────── */
.msg-user {
    display:flex; justify-content:flex-end; margin-bottom:1.1rem;
}
.bubble-user {
    background: linear-gradient(135deg, #FC6D26 0%, #FCA326 100%);
    color:#fff; padding:.72rem 1.15rem;
    border-radius:18px 18px 4px 18px;
    max-width:72%; font-size:.92rem; line-height:1.55;
    box-shadow: 0 2px 10px rgba(252,109,38,.28);
    word-wrap: break-word;
}
.msg-bot {
    display:flex; gap:.7rem; margin-bottom:1.1rem; align-items:flex-start;
}
.bot-av {
    width:34px; height:34px; flex-shrink:0;
    background: linear-gradient(135deg,#16141C,#2A2635);
    border:2px solid rgba(252,109,38,.28);
    border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1rem;
}
.bubble-bot {
    background:#fff;
    border:1px solid #E8E8E8;
    border-radius:4px 18px 18px 18px;
    padding:1rem 1.2rem;
    max-width:88%;
    font-size:.9rem; line-height:1.65;
    box-shadow:0 1px 6px rgba(0,0,0,.05);
    word-wrap: break-word;
}

/* ── Pipeline pill ────────────────────────────────────────── */
.pipe-pill {
    display:inline-flex; align-items:center; gap:.35rem;
    padding:.18rem .65rem; border-radius:999px;
    font-size:.68rem; font-weight:700; margin:.6rem 0 .2rem;
}
.pipe-simple { background:rgba(63,185,80,.1); color:#2DA44E; border:1px solid rgba(63,185,80,.22); }
.pipe-full   { background:rgba(155,127,224,.1); color:#9B7FE0; border:1px solid rgba(155,127,224,.22); }

/* ── Sources ──────────────────────────────────────────────── */
.sources-wrap { margin-top:.9rem; border-top:1px solid #F0F0F0; padding-top:.7rem; }
.sources-label {
    font-size:.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.09em; color:#BBB; margin-bottom:.4rem;
}
.src-chip {
    display:inline-flex; align-items:center; gap:.25rem;
    padding:.2rem .65rem; background:#F7F7F7; border:1px solid #E5E5E5;
    border-radius:6px; font-size:.73rem; color:#555;
    margin:.2rem .2rem .2rem 0;
    text-decoration:none; transition: all .14s;
}
.src-chip:hover {
    background:rgba(252,109,38,.06);
    border-color:rgba(252,109,38,.3);
    color:#FC6D26; text-decoration:none;
}

/* ── Response meta ────────────────────────────────────────── */
.resp-meta { font-size:.68rem; color:#CCC; margin-top:.4rem; }

/* ── Sidebar ──────────────────────────────────────────────── */
.sb-section {
    background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.08);
    border-radius:10px;
    padding:.9rem 1rem;
    margin-bottom:.9rem;
}
.sb-title {
    font-size:.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.1em; color:#555; margin-bottom:.6rem;
}
.sb-tech-item {
    display:flex; align-items:center; gap:.5rem;
    padding:.32rem 0; border-bottom:1px solid rgba(255,255,255,.05);
    font-size:.82rem;
}
.sb-tech-item:last-child { border-bottom:none; }
.sb-tech-name { font-weight:600; color:#DDD; flex:1; }
.sb-tech-tag {
    font-size:.65rem; font-weight:700; padding:.1rem .45rem;
    border-radius:999px; background:rgba(63,185,80,.14);
    color:#3FB950; border:1px solid rgba(63,185,80,.22);
}
.sb-metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:.5rem; }
.sb-metric {
    background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08);
    border-radius:8px; padding:.6rem; text-align:center;
}
.sb-metric-val { font-size:1.35rem; font-weight:800; color:#FC6D26; }
.sb-metric-lbl { font-size:.62rem; color:#666; text-transform:uppercase; letter-spacing:.05em; }

/* ── Chat input tweaks ────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#333; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#FC6D26; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_crew():
    """Create the CrewAI crew once; cached for the lifetime of the server."""
    return create_simple_retrieval_crew()


def _extract_sources(text: str) -> list[dict]:
    """Pull GitLab docs URLs from the answer text and return unique cards."""
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
    """Run the async parallel pre-process in a fresh thread to avoid event-loop conflicts."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, run_parallel_preprocess(query))
        try:
            return future.result(timeout=45)
        except Exception:
            return ""


def _ask(query: str) -> dict:
    """
    Route query, run the crew, return a result dict with answer + metadata.
    """
    start = time.time()
    use_full = _needs_full_pipeline(query)

    if use_full:
        preprocess_context = _run_async_preprocess(query)
        enriched = f"{query}\n\n[PREPROCESSING CONTEXT]\n{preprocess_context}" if preprocess_context else query
    else:
        enriched = query

    crew = _get_crew()
    result = crew.kickoff(inputs={"user_query": enriched})
    answer = result.raw if hasattr(result, "raw") else str(result)

    return {
        "answer": answer,
        "pipeline": "full" if use_full else "simple",
        "elapsed": round(time.time() - start, 1),
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
        "<div style='padding:.6rem 0 1.2rem'>"
        "<div style='font-size:1.6rem;margin-bottom:.2rem'>🦊</div>"
        "<div style='font-size:1rem;font-weight:800;color:#FC6D26;'>GitLab AI</div>"
        "<div style='font-size:.75rem;color:#555;'>Documentation Expert</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Architecture ──────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="sb-section">
  <div class="sb-title">🏗 Pipeline Architecture</div>
  <div style="font-family:ui-monospace,monospace;font-size:.72rem;color:#888;line-height:1.85;">
    <span style="color:#FC6D26;font-weight:700;">User Query</span><br>
    &nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#9B7FE0;">Smart Router</span><br>
    <span style="color:#555;font-size:.65rem;">&nbsp;&nbsp;(clear q → simple | vague q → full)</span><br>
    &nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#9B7FE0;">Intent+Rewrite</span>&nbsp;<span style="color:#555;">[async∥]</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#FC6D26;font-weight:700;">Retriever Agent</span><br>
    <span style="color:#555;font-size:.65rem;">&nbsp;&nbsp;ChromaDB · cosine · RRF</span><br>
    &nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#3FB950;font-weight:700;">Synthesiser Agent</span><br>
    <span style="color:#555;font-size:.65rem;">&nbsp;&nbsp;grounded · cited · no hallucination</span><br>
    &nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#FC6D26;font-weight:700;">Answer + Sources</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="sb-section">
  <div class="sb-title">⚡ Tech Stack</div>
  <div class="sb-tech-item"><span class="sb-tech-name">CrewAI</span><span class="sb-tech-tag">Agents</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">ChromaDB</span><span class="sb-tech-tag">Vector DB</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">BAAI/bge-small</span><span class="sb-tech-tag">Embeddings</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">OpenRouter</span><span class="sb-tech-tag">LLM</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">LangChain</span><span class="sb-tech-tag">RAG</span></div>
  <div class="sb-tech-item"><span class="sb-tech-name">asyncio</span><span class="sb-tech-tag">Parallel</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="sb-section">
  <div class="sb-title">📊 System Stats</div>
  <div class="sb-metric-grid">
    <div class="sb-metric"><div class="sb-metric-val">$0</div><div class="sb-metric-lbl">Total Cost</div></div>
    <div class="sb-metric"><div class="sb-metric-val">8K+</div><div class="sb-metric-lbl">Docs Indexed</div></div>
    <div class="sb-metric"><div class="sb-metric-val">384</div><div class="sb-metric-lbl">Embed Dims</div></div>
    <div class="sb-metric"><div class="sb-metric-val">0.40</div><div class="sb-metric-lbl">Min Relevance</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Sample questions ──────────────────────────────────────────────────────
    st.markdown(
        "<div class='sb-title' style='margin-top:.5rem;padding-left:.2rem'>💡 Try asking</div>",
        unsafe_allow_html=True,
    )
    sample_questions = [
        "How do I cache npm packages in GitLab CI?",
        "What is the difference between stages and jobs?",
        "How do I set up a Docker-in-Docker pipeline?",
        "How do I protect a branch in GitLab?",
        "What are GitLab CI/CD variables and how do I use them?",
        "How do I run SAST security scanning in my pipeline?",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_query = q

    st.markdown(
        "<div style='margin-top:1.5rem;font-size:.65rem;color:#444;text-align:center;'>"
        "Built by Raghav Gupta · Agentic RAG<br>"
        "CrewAI · ChromaDB · sentence-transformers"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — HERO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CARDS — shown when chat is empty
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown(
        """
<div class="features-grid">

  <div class="fc">
    <span class="fc-icon">🤖</span>
    <div class="fc-title">Multi-Agent Architecture</div>
    <div class="fc-desc">
      CrewAI orchestrates a <strong>Retriever</strong> and <strong>Synthesiser</strong> agent
      in sequence. Vague queries additionally trigger parallel
      Intent Classification and Query Rewriting.
    </div>
    <div class="fc-chips">
      <span class="chip">CrewAI</span>
      <span class="chip">asyncio</span>
      <span class="chip">parallel</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">🔍</span>
    <div class="fc-title">Semantic Vector Search</div>
    <div class="fc-desc">
      8,000+ GitLab docs chunked, embedded locally, and stored in ChromaDB.
      Cosine similarity search with <strong>Reciprocal Rank Fusion</strong>
      for multi-query retrieval.
    </div>
    <div class="fc-chips">
      <span class="chip">ChromaDB</span>
      <span class="chip">RRF</span>
      <span class="chip">cosine</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">📄</span>
    <div class="fc-title">Citation-Grounded Answers</div>
    <div class="fc-desc">
      Every fact in the answer is backed by a source URL from
      <code>docs.gitlab.com</code>. Deprecated features are
      flagged. No hallucinations.
    </div>
    <div class="fc-chips">
      <span class="chip">RAG</span>
      <span class="chip">no hallucination</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">⚡</span>
    <div class="fc-title">Smart Query Routing</div>
    <div class="fc-desc">
      A lightweight router decides: clear questions go straight to retrieval
      (~5s); vague or short queries run async intent + rewrite first,
      cutting pre-processing time in half.
    </div>
    <div class="fc-chips">
      <span class="chip">routing logic</span>
      <span class="chip">async∥</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">🧠</span>
    <div class="fc-title">Zero-Cost Local Embeddings</div>
    <div class="fc-desc">
      <code>BAAI/bge-small-en-v1.5</code> runs entirely on CPU — no GPU,
      no API fees. Embeddings are disk-cached so re-indexing skips
      already-processed chunks.
    </div>
    <div class="fc-chips">
      <span class="chip">sentence-transformers</span>
      <span class="chip">384-dim</span>
      <span class="chip">$0</span>
    </div>
  </div>

  <div class="fc">
    <span class="fc-icon">🏗️</span>
    <div class="fc-title">Production-Grade Ingestion</div>
    <div class="fc-desc">
      Git sparse-checkout downloads only the <code>/doc</code> folder (~80 MB
      vs 4 GB full repo). Header-aware Markdown chunking preserves context
      across 30,000+ chunks.
    </div>
    <div class="fc-chips">
      <span class="chip">sparse-checkout</span>
      <span class="chip">LangChain</span>
      <span class="chip">tiktoken</span>
    </div>
  </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # Architecture flow callout
    st.markdown(
        """
<div class="arch-callout">
  <span class="hl">Query</span>
  <span class="dim"> → </span>
  <span class="hl2">Router</span>
  <span class="dim"> → </span>
  <span class="hl2">[Intent + Rewrite]</span>
  <span class="dim"> (parallel, only for vague queries) → </span>
  <span class="hl">Retriever Agent</span>
  <span class="dim"> → semantic_search / filtered_search / multi_query_search → </span>
  <span class="hl3">Heuristic Validator</span>
  <span class="dim"> (relevance ≥ 0.40) → </span>
  <span class="hl">Synthesiser Agent</span>
  <span class="dim"> → </span>
  <span class="hl3">Cited Answer</span>
</div>
""",
        unsafe_allow_html=True,
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
        pipe_class = "pipe-full" if msg.get("pipeline") == "full" else "pipe-simple"
        pipe_label = (
            "🔄 Full pipeline — intent + rewrite + retrieve + synthesise"
            if msg.get("pipeline") == "full"
            else "⚡ Simple pipeline — retrieve + synthesise"
        )
        sources_html = ""
        if msg.get("sources"):
            chips = "".join(
                f'<a class="src-chip" href="{s["url"]}" target="_blank">📎 {s["label"]}</a>'
                for s in msg["sources"]
            )
            sources_html = (
                f'<div class="sources-wrap">'
                f'<div class="sources-label">Sources</div>'
                f'{chips}'
                f'</div>'
            )

        st.markdown(
            f'<div class="msg-bot">'
            f'  <div class="bot-av">🦊</div>'
            f'  <div style="flex:1;min-width:0;">'
            f'    <div class="bubble-bot">{msg["content"]}</div>'
            f'    <div class="pipe-pill {pipe_class}">{pipe_label}</div>'
            f'    {sources_html}'
            f'    <div class="resp-meta">⏱ {msg.get("elapsed", "?")}s</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────

# Check if a sample question button was clicked
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = ""
else:
    query = st.chat_input("Ask anything about GitLab…")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Show user bubble immediately
    st.markdown(
        f'<div class="msg-user"><div class="bubble-user">{query}</div></div>',
        unsafe_allow_html=True,
    )

    # Run the agents with a status widget
    with st.status("🤖 Agents are working…", expanded=True) as status:
        route = "full pipeline (intent + rewrite → retrieve → synthesise)" if _needs_full_pipeline(query) else "simple pipeline (retrieve → synthesise)"
        st.write(f"**Route:** {route}")
        st.write("Searching 8,000+ GitLab documentation chunks…")

        try:
            result = _ask(query)
            status.update(label="✅ Done!", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="❌ Error", state="error", expanded=True)
            st.error(f"Something went wrong: {exc}")
            result = None

    if result:
        pipe_class = "pipe-full" if result["pipeline"] == "full" else "pipe-simple"
        pipe_label = (
            "🔄 Full pipeline — intent + rewrite + retrieve + synthesise"
            if result["pipeline"] == "full"
            else "⚡ Simple pipeline — retrieve + synthesise"
        )
        sources_html = ""
        if result["sources"]:
            chips = "".join(
                f'<a class="src-chip" href="{s["url"]}" target="_blank">📎 {s["label"]}</a>'
                for s in result["sources"]
            )
            sources_html = (
                f'<div class="sources-wrap">'
                f'<div class="sources-label">Sources</div>'
                f'{chips}'
                f'</div>'
            )

        answer_html = result["answer"].replace("<", "&lt;").replace(">", "&gt;")

        st.markdown(
            f'<div class="msg-bot">'
            f'  <div class="bot-av">🦊</div>'
            f'  <div style="flex:1;min-width:0;">'
            f'    <div class="bubble-bot">{answer_html}</div>'
            f'    <div class="pipe-pill {pipe_class}">{pipe_label}</div>'
            f'    {sources_html}'
            f'    <div class="resp-meta">⏱ {result["elapsed"]}s</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Save to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_html,
            "pipeline": result["pipeline"],
            "elapsed": result["elapsed"],
            "sources": result["sources"],
        })

    st.rerun()
