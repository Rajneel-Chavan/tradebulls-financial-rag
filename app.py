"""
Tradebulls Financial Intelligence RAG — Streamlit UI

Features:
  - Session-isolated conversations
  - PDF upload to knowledge base
  - Source type filtering
  - Route indicator showing which pipeline was used
  - Confidence score from grading
  - Financial disclaimer on market responses
  - Chat history with export
"""

import os
import json
import streamlit as st
from datetime import datetime

from backend.config import PDF_DIRECTORY
from backend.data_loader import load_all_sources
from backend.chunker import chunk_documents
from backend.vector_store import (
    create_vector_store,
    create_session_id,
    get_existing_vector_store,
    list_sessions,
    delete_session,
)
from backend.rag_graph import run_query


# ── Page Config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tradebulls Financial Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .route-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .route-retrieve { background: #1a3a5c; color: #4da6ff; }
    .route-direct { background: #1a3c2a; color: #4dff88; }
    .route-factcheck { background: #3c2a1a; color: #ff9944; }
    .source-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 11px;
        margin: 2px;
        background: #1e2530;
        color: #8899aa;
    }
    .score-box {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        background: #162218;
        color: #66cc88;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = create_session_id()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Tradebulls Intelligence")
    st.markdown(f"**Session:** `{st.session_state.session_id}`")
    st.markdown("---")

    # Data Loading
    st.markdown("### 📊 Knowledge Base")

    if st.button("🔄 Load Tradebulls Data", use_container_width=True):
        with st.spinner("Loading all data sources..."):
            documents = load_all_sources()
            chunks = chunk_documents(documents)
            vector_store = create_vector_store(
                chunks, st.session_state.session_id
            )
            st.session_state.vector_store = vector_store
            st.session_state.all_chunks = chunks
            st.session_state.data_loaded = True
            st.success(f"Indexed {len(chunks)} chunks from 4 sources")

    # PDF Upload
    uploaded_file = st.file_uploader(
        "Upload additional PDF",
        type=["pdf"],
        help="Upload Tradebulls research reports",
    )

    if uploaded_file:
        pdf_path = os.path.join(PDF_DIRECTORY, uploaded_file.name)
        os.makedirs(PDF_DIRECTORY, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"Saved: {uploaded_file.name}")

    st.markdown("---")

    # Data Status
    if st.session_state.data_loaded:
        st.markdown("### 📁 Indexed Data")
        st.markdown(f"**Chunks:** {len(st.session_state.all_chunks)}")

        source_counts = {}
        for chunk in st.session_state.all_chunks:
            src = chunk.metadata.get("source_type", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        for src, count in source_counts.items():
            st.markdown(f"- {src}: **{count}** chunks")

    st.markdown("---")

    # Session Management
    st.markdown("### 🗂️ Sessions")
    if st.button("➕ New Session", use_container_width=True):
        st.session_state.session_id = create_session_id()
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.all_chunks = []
        st.session_state.data_loaded = False
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Export Chat
    if st.session_state.chat_history:
        chat_export = json.dumps(
            st.session_state.chat_history, indent=2, default=str
        )
        st.download_button(
            "📥 Export Chat (.json)",
            data=chat_export,
            file_name=f"tradebulls_chat_{st.session_state.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown(
        "Built with LangGraph · HyDE · RRF · Cohere · RAGAS",
        help="Advanced RAG pipeline with agentic routing",
    )


# ── Main Chat Area ────────────────────────────────────────────────────

st.markdown("# 📈 Tradebulls Financial Intelligence")
st.markdown(
    "Advanced RAG system with agentic routing, HyDE query expansion, "
    "RRF fusion retrieval, and Cohere cross-encoder reranking."
)

if not st.session_state.data_loaded:
    st.info(
        "👈 Click **Load Tradebulls Data** in the sidebar to index "
        "all data sources before querying."
    )

# Display Chat History
for msg in st.session_state.chat_history:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

        if role == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            cols = st.columns(3)

            with cols[0]:
                route = meta.get("route", "")
                route_class = {
                    "retrieve": "route-retrieve",
                    "direct_answer": "route-direct",
                    "financial_fact_check": "route-factcheck",
                }.get(route, "route-retrieve")

                st.markdown(
                    f'<span class="route-badge {route_class}">'
                    f"🔀 {route}</span>",
                    unsafe_allow_html=True,
                )

            with cols[1]:
                sources = meta.get("sources_used", [])
                source_html = " ".join(
                    f'<span class="source-tag">{s}</span>' for s in sources
                )
                st.markdown(source_html, unsafe_allow_html=True)

            with cols[2]:
                retries = meta.get("retry_count", 0)
                grounded = meta.get("is_grounded", False)
                st.markdown(
                    f'<span class="score-box">'
                    f"✅ Grounded" if grounded else "⚠️ Unverified"
                    f" | Retries: {retries}</span>",
                    unsafe_allow_html=True,
                )


# ── Query Input ───────────────────────────────────────────────────────

# Suggestion pills
if not st.session_state.chat_history and st.session_state.data_loaded:
    st.markdown("**Try asking:**")
    suggestions = [
        "What does the latest Tradebulls report say about Nifty?",
        "What are the key support and resistance levels?",
        "How do I open a demat account?",
        "What trading platforms does Tradebulls offer?",
    ]

    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.pending_query = suggestion
            st.rerun()

# Chat input
query = st.chat_input(
    "Ask about Tradebulls reports, market analysis, or trading..."
)

# Handle suggestion click
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query

if query:
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Run agentic RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing..."):
            result = run_query(
                query=query,
                session_id=st.session_state.session_id,
                vector_store=st.session_state.vector_store,
                all_chunks=st.session_state.all_chunks,
            )

            generation = result.get("generation", "No response generated.")
            st.markdown(generation)

            # Show metadata
            metadata = {
                "route": result.get("route", ""),
                "sources_used": result.get("sources_used", []),
                "retry_count": result.get("retry_count", 0),
                "is_grounded": result.get("is_grounded", False),
                "doc_relevance_score": result.get("doc_relevance_score", 0),
            }

            cols = st.columns(3)
            with cols[0]:
                route = metadata["route"]
                route_class = {
                    "retrieve": "route-retrieve",
                    "direct_answer": "route-direct",
                    "financial_fact_check": "route-factcheck",
                }.get(route, "route-retrieve")
                st.markdown(
                    f'<span class="route-badge {route_class}">'
                    f"🔀 {route}</span>",
                    unsafe_allow_html=True,
                )

            with cols[1]:
                sources = metadata["sources_used"]
                source_html = " ".join(
                    f'<span class="source-tag">{s}</span>' for s in sources
                )
                st.markdown(source_html, unsafe_allow_html=True)

            with cols[2]:
                grounded_text = "✅ Grounded" if metadata["is_grounded"] else "⚠️ Unverified"
                st.markdown(
                    f'<span class="score-box">'
                    f"{grounded_text} | Retries: {metadata['retry_count']}</span>",
                    unsafe_allow_html=True,
                )

    # Save assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": generation,
        "metadata": metadata,
    })
