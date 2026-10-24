"""ResearchMind AI — Streamlit research workspace.

Run from the project root: streamlit run app.py
"""
import hashlib
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.btw_handler import handle_btw
from backend.paper_loader import (
    load_arxiv,
    load_document,
    load_semantic_scholar,
    load_webpage,
    search_semantic_scholar_papers,
)
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers

st.set_page_config(
    page_title="ResearchMind AI | Agentic Research Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Appearance only. The application works without custom fonts or external CSS.
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1220; color: #e8eef8; }
[data-testid="stSidebar"] { background: #111c2e; border-right: 1px solid #29384d; }
[data-testid="stSidebar"] .stButton > button { border-radius: 10px; }
.block-container { max-width: 1160px; padding-top: 2rem; }
.rm-eyebrow { color:#67d6c0; font-size:.75rem; letter-spacing:.16em; font-weight:750; text-transform:uppercase; }
.rm-hero { padding:1.6rem 1.8rem; border:1px solid #2b425b; border-radius:18px;
           background:linear-gradient(120deg,#132a42 0%,#111c30 60%,#152139 100%); margin-bottom:1.2rem; }
.rm-hero h1 { margin:.25rem 0; font-size:2.25rem; line-height:1.2; color:#f6f9ff; }
.rm-hero p { margin:.5rem 0 0; color:#c0d1e5; }
.rm-tagline { color:#77dfce !important; font-weight:650; }
.rm-chip { display:inline-block; padding:.32rem .65rem; margin:.55rem .4rem 0 0;
           background:#1b3147; color:#d3e5f4; border:1px solid #34516b;
           border-radius:999px; font-size:.78rem; }
.rm-note { color:#a8bbce; font-size:.85rem; }
[data-testid="stChatMessage"] { border:1px solid #26394e; border-radius:14px; background:#121e30; }
</style>
""", unsafe_allow_html=True)

SESSIONS_FILE = Path("sessions.json")


@st.cache_resource
def get_graph():
    return build_graph()


@st.cache_resource
def get_naming_llm():
    return ChatOpenAI(model="gpt-5-mini")


def load_sessions() -> dict:
    try:
        value = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    result = {}
    for key, value in values.items():
        if key == "messages":
            result[key] = [
                {"type": type(msg).__name__, "content": str(msg.content)[:300]}
                for msg in (value or [])
            ]
        elif key == "retrieved_docs":
            result[key] = [
                {"content": doc.page_content[:300], "metadata": doc.metadata}
                for doc in (value or [])
            ]
        else:
            result[key] = value
    return result


def generate_session_name(first_message: str) -> str:
    try:
        response = get_naming_llm().invoke([
            {"role": "system", "content": "Create a concise 3–5 word title for this research conversation. Return only the title."},
            {"role": "user", "content": first_message[:500]},
        ])
        return str(response.content).strip().strip('"')[:70] or "New Research"
    except Exception:
        return "New Research"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    meta = st.session_state.sessions_meta.get(session_id)
    if not meta or meta.get("is_named"):
        return
    meta["name"] = generate_session_name(first_message)
    meta["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid, "name": "New Research", "created_at": datetime.now().isoformat(), "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    """Restore visible turns from checkpoint messages, excluding internal tool calls."""
    try:
        state = graph.get_state({"configurable": {"thread_id": session_id}})
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if name == "HumanMessage":
                # Internal rewritten retrieval queries are not separate user turns.
                if chats and chats[-1]["role"] == "user":
                    continue
                chats.append({"role": "user", "content": content})
            elif name in ("AIMessage", "AIMessageChunk") and content and not getattr(msg, "tool_calls", None):
                # Internal retrieval-agent narration is not a final assistant answer.
                if chats and chats[-1]["role"] == "user":
                    turn += 1
                    chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        st.session_state.turns[session_id] = sum(
            item["role"] == "assistant" for item in st.session_state.chats[session_id]
        )


def ingest(docs, session_id: str, label: str) -> None:
    if not docs:
        raise ValueError("No readable content was found in the document.")
    add_paper(docs, session_id)
    st.toast(f"Added to knowledge base: {label}", icon="✅")


def paper_label(paper: dict) -> str:
    title = paper.get("title") or "Untitled paper"
    year = paper.get("year") or "Year unknown"
    return f"{title} ({year})"


graph = get_graph()
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(st.session_state.sessions_meta.values(), key=lambda s: s.get("created_at", ""))
        switch_session(latest["id"])
    else:
        st.session_state.active_session_id = create_session()
active_sid = st.session_state.active_session_id

# Sidebar navigation and knowledge-base tools.
with st.sidebar:
    st.markdown("### 🧠 ResearchMind AI")
    st.caption("Your agentic research workspace")
    if st.button("＋ New Research", use_container_width=True, type="primary"):
        st.session_state.active_session_id = create_session()
        st.rerun()

    with st.expander("💬 Research sessions", expanded=True):
        sorted_sessions = sorted(
            st.session_state.sessions_meta.values(),
            key=lambda item: item.get("created_at", ""), reverse=True,
        )
        for session in sorted_sessions:
            sid = session["id"]
            if st.button(
                ("● " if sid == active_sid else "") + session.get("name", "New Research"),
                key=f"session_{sid}",
                use_container_width=True,
                type="primary" if sid == active_sid else "secondary",
            ):
                if sid != active_sid:
                    switch_session(sid)
                    st.rerun()

    st.divider()
    st.markdown("### 📚 Knowledge base")
    source_tab, papers_tab = st.tabs(["Add sources", "Loaded papers"])
    with source_tab:
        with st.expander("📁 Upload documents", expanded=True):
            files = st.file_uploader(
                "PDF, TXT, Markdown", type=["pdf", "txt", "md", "markdown"],
                accept_multiple_files=True, key=f"files_{active_sid}",
            )
            if st.button("Add uploaded files", key=f"add_files_{active_sid}", use_container_width=True):
                if not files:
                    st.warning("Choose at least one file.")
                else:
                    for file in files:
                        signature = hashlib.sha256(file.getvalue()).hexdigest()
                        known = st.session_state.setdefault(f"uploaded_{active_sid}", set())
                        if signature in known:
                            st.info(f"Already added in this browser session: {file.name}")
                            continue
                        path = None
                        try:
                            with st.spinner(f"Reading {file.name}…"):
                                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                                    tmp.write(file.getvalue())
                                    path = tmp.name
                                docs = load_document(path)
                                for doc in docs:
                                    doc.metadata.update({"title": Path(file.name).stem, "source": file.name, "source_type": "upload"})
                                ingest(docs, active_sid, file.name)
                                known.add(signature)
                        except Exception as exc:
                            st.error(f"{file.name}: {exc}")
                        finally:
                            if path:
                                Path(path).unlink(missing_ok=True)
        with st.expander("🔗 Web pages"):
            urls_text = st.text_area("One URL per line", key=f"urls_{active_sid}", height=85)
            if st.button("Add web pages", key=f"add_urls_{active_sid}", use_container_width=True):
                urls = [line.strip() for line in urls_text.splitlines() if line.strip()]
                if not urls:
                    st.warning("Enter a URL.")
                for url in urls:
                    if not url.startswith(("https://", "http://")):
                        st.error(f"Invalid URL: {url}")
                        continue
                    try:
                        with st.spinner("Reading web page…"):
                            ingest(load_webpage(url), active_sid, url[:60])
                    except Exception as exc:
                        st.error(f"{url}: {exc}")
        with st.expander("📄 arXiv"):
            arxiv_query = st.text_input(
                "Paper title or arXiv ID", placeholder="1706.03762", key=f"arxiv_{active_sid}",
            )
            if st.button("Add arXiv paper", key=f"add_arxiv_{active_sid}", use_container_width=True):
                if not arxiv_query.strip():
                    st.warning("Enter a title or arXiv ID.")
                else:
                    try:
                        with st.spinner("Downloading arXiv PDF…"):
                            docs = load_arxiv(arxiv_query.strip())
                            ingest(docs, active_sid, docs[0].metadata.get("title", arxiv_query))
                    except Exception as exc:
                        st.error(f"arXiv: {exc}")
        with st.expander("🎓 Semantic Scholar", expanded=True):
            s2_query = st.text_input(
                "Find academic papers", placeholder="Agentic retrieval augmented generation",
                key=f"s2_query_{active_sid}",
            )
            s2_limit = st.slider("Number of results", 1, 10, 5, key=f"s2_limit_{active_sid}")
            if st.button("Search Semantic Scholar", key=f"search_s2_{active_sid}", use_container_width=True):
                if not s2_query.strip():
                    st.warning("Enter a paper title or topic.")
                else:
                    try:
                        with st.spinner("Searching academic literature…"):
                            st.session_state[f"s2_results_{active_sid}"] = search_semantic_scholar_papers(
                                s2_query.strip(), limit=s2_limit,
                            )
                    except Exception as exc:
                        st.error(f"Semantic Scholar: {exc}")
            results = st.session_state.get(f"s2_results_{active_sid}", [])
            if results:
                st.caption(f"{len(results)} results. Select a paper to see its details.")
                valid_results = [p for p in results if p.get("paperId")]
                if valid_results:
                    selected_id = st.selectbox(
                        "Select paper", options=[p["paperId"] for p in valid_results],
                        format_func=lambda pid: paper_label(next(p for p in valid_results if p["paperId"] == pid)),
                        key=f"selected_s2_{active_sid}",
                    )
                    selected = next(p for p in valid_results if p["paperId"] == selected_id)
                    st.markdown(f"**{selected.get('title', 'Untitled')}**")
                    st.caption(
                        f"{selected.get('year') or 'Year unknown'} · "
                        f"{selected.get('citationCount') or 0} citations · "
                        f"{', '.join(a.get('name', '') for a in (selected.get('authors') or [])[:3])}"
                    )
                    if selected.get("abstract"):
                        st.caption(selected["abstract"][:700] + ("…" if len(selected["abstract"]) > 700 else ""))
                    if selected.get("url"):
                        st.link_button("View paper", selected["url"], use_container_width=True)
                    if not (selected.get("openAccessPdf") or {}).get("url"):
                        st.info("No open-access PDF listed. You can view metadata, but full-text ingestion is unavailable.")
                    if st.button(
                        "Add selected PDF to knowledge base", key=f"ingest_s2_{active_sid}",
                        use_container_width=True,
                        disabled=not bool((selected.get("openAccessPdf") or {}).get("url")),
                    ):
                        try:
                            with st.spinner("Downloading and indexing selected paper…"):
                                docs = load_semantic_scholar(selected.get("title", ""), paper_id=selected_id)
                                ingest(docs, active_sid, selected.get("title", "paper"))
                        except Exception as exc:
                            st.error(f"Could not ingest paper: {exc}")
            elif f"s2_results_{active_sid}" in st.session_state:
                st.info("No matching papers found.")
    with papers_tab:
        try:
            titles = list_papers(active_sid)
            if titles:
                st.caption(f"{len(titles)} document(s) in this research session")
                for title in titles:
                    st.markdown(f"• {title}")
            else:
                st.caption("No documents added to this session yet.")
        except Exception as exc:
            st.warning(f"Unable to list documents: {exc}")

# Main research workspace.
st.markdown("""
<div class="rm-hero">
  <div class="rm-eyebrow">Agentic Research Intelligence Assistant</div>
  <h1>🧠 ResearchMind AI</h1>
  <p class="rm-tagline">Explore Research. Validate Knowledge. Discover Insights.</p>
  <p>Analyze research documents, discover academic papers, verify claims against current sources,
     and continue research in isolated conversations.</p>
  <span class="rm-chip">📚 Document RAG</span>
  <span class="rm-chip">🎓 Semantic Scholar</span>
  <span class="rm-chip">📄 arXiv</span>
  <span class="rm-chip">✓ Claim verification</span>
  <span class="rm-chip">🌐 Live web search</span>
</div>
""", unsafe_allow_html=True)

st.caption(f"Current workspace: {st.session_state.sessions_meta.get(active_sid, {}).get('name', 'New Research')}")
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("graph_state"):
            with st.expander(f"Graph details · turn {msg.get('turn', '?')}"):
                st.json(msg["graph_state"])

prompt = st.chat_input("Ask about papers, verify a research claim, or discover new findings…")
if prompt:
    is_btw = prompt.strip().lower().startswith("/btw")
    if is_btw:
        query = prompt.strip()[4:].strip()
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Private side question — not stored in the research conversation")
        with st.chat_message("assistant"):
            if not query:
                st.info("Try `/btw What is a transformer model?`")
            else:
                try:
                    placeholder = st.empty()
                    response_text = ""
                    for chunk in handle_btw(query):
                        response_text += chunk
                        placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                except Exception as exc:
                    st.error(f"Side question failed: {exc}")
            st.caption("Not saved to research conversation")
    else:
        chats = st.session_state.chats.setdefault(active_sid, [])
        is_first_message = not chats
        st.session_state.turns[active_sid] = st.session_state.turns.get(active_sid, 0) + 1
        turn = st.session_state.turns[active_sid]
        with st.chat_message("user"):
            st.markdown(prompt)
        chats.append({"role": "user", "content": prompt})
        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""
            try:
                for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                    if metadata.get("langgraph_node") == "generate_answer" and getattr(chunk, "content", None):
                        content = chunk.content
                        if isinstance(content, str):
                            response_text += content
                            placeholder.markdown(response_text + "▌")
                final_values = graph.get_state(config).values
                # The stored final answer is authoritative; avoids duplicate tokens when
                # the graph emits both LLM chunks and a final AIMessage.
                response_text = final_values.get("answer") or response_text or "No response generated."
                placeholder.markdown(response_text)
                snapshot = _serialize_state(final_values)
                with st.expander(f"Graph details · turn {turn}"):
                    st.json(snapshot)
                chats.append({"role": "assistant", "content": response_text, "turn": turn, "graph_state": snapshot})
                if is_first_message:
                    maybe_rename_session(active_sid, prompt)
                    st.rerun()
            except Exception as exc:
                st.error(f"Research request failed: {exc}")
                chats.append({"role": "assistant", "content": "I couldn't complete that request. Please retry.", "turn": turn, "graph_state": {}})
