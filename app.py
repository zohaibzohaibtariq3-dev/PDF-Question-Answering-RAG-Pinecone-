"""Streamlit UI for the Pinecone PDF RAG system."""
from __future__ import annotations

import streamlit as st

from rag.config import get_settings
from rag.exceptions import EmptyQueryError, RAGError
from rag.pipeline import Answer, RAGPipeline
from rag.query_logger import setup_logging

st.set_page_config(page_title="PDF RAG · Pinecone", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and connecting to Pinecone…")
def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    setup_logging(settings.log_dir)
    return RAGPipeline(settings)


try:
    pipeline = get_pipeline()
except RAGError as exc:
    st.error(f"Setup problem: {exc}")
    st.stop()

# ----------------------------------------------------------------- session state
st.session_state.setdefault("history", [])  # list[Answer]
st.session_state.setdefault("doc_names", {})  # namespace -> original file name
st.session_state.setdefault("selected_ns", [])


def refresh_documents() -> dict[str, int]:
    try:
        return pipeline.list_documents()
    except RAGError as exc:
        st.error(f"Pinecone error: {exc}")
        return {}


# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, 50,
                           help="Applied the next time you index a PDF.")
    overlap = st.slider("Chunk overlap", 0, 400, 100, 25)
    overlap = min(overlap, chunk_size // 2)
    top_k = st.slider("Top-k chunks", 1, 10, 4)
    threshold = st.slider("Similarity threshold (cosine)", 0.0, 1.0, 0.05, 0.05,
                          help="Chunks scoring below this are discarded.")
    use_pages = st.checkbox("Filter by page range")
    page_range = None
    if use_pages:
        c1, c2 = st.columns(2)
        lo = c1.number_input("From page", min_value=1, value=1, step=1)
        hi = c2.number_input("To page", min_value=1, value=10, step=1)
        page_range = (int(lo), int(hi))

    st.divider()
    st.subheader("🕘 Query history")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            icon = "✅" if item.answered else "⚠️"
            st.caption(f"{icon} {item.question[:70]}")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No questions yet.")
    st.divider()
    st.caption(f"Embeddings: `{pipeline.settings.embedding_model.split('/')[-1]}`")
    st.caption(f"LLM: `{pipeline.settings.groq_model}`")
    st.caption(f"Pinecone index: `{pipeline.settings.pinecone_index}`")

# ------------------------------------------------------------------------ main
st.title("📄 PDF Question Answering (RAG + Pinecone)")
st.caption("Answers come strictly from your PDFs, with page numbers, excerpts and similarity scores.")

st.subheader("1 · Upload & index")
uploads = st.file_uploader("Upload one or more PDFs (max 20 MB each)", type=["pdf"],
                           accept_multiple_files=True)
if st.button("Process & index", type="primary", disabled=not uploads):
    for up in uploads:
        with st.status(f"Indexing {up.name}…", expanded=False) as status:
            try:
                result = pipeline.ingest(up.getvalue(), up.name, chunk_size, overlap)
            except RAGError as exc:
                status.update(label=f"❌ {up.name}", state="error")
                st.error(str(exc))
                continue
            st.session_state.doc_names[result.namespace] = result.doc_name
            if result.namespace not in st.session_state.selected_ns:
                st.session_state.selected_ns.append(result.namespace)
            status.update(
                label=f"✅ {result.doc_name}: {result.pages} pages → {result.chunks} chunks "
                      f"in {result.seconds}s", state="complete")

docs = refresh_documents()
st.subheader("2 · Ask questions")
if not docs:
    st.info("No documents indexed yet. Upload a PDF above to get started.")
    selected = []
else:
    st.session_state.selected_ns = [n for n in st.session_state.selected_ns if n in docs]
    selected = st.multiselect(
        "Search in", options=list(docs), key="selected_ns",
        format_func=lambda ns: f"{st.session_state.doc_names.get(ns, ns)}  ({docs[ns]} chunks)")
    if selected and st.button("🗑️ Delete selected document(s) from Pinecone"):
        for ns in selected:
            try:
                pipeline.delete_document(ns)
            except RAGError as exc:
                st.error(str(exc))
        st.rerun()


def render_answer(a: Answer, key: str) -> None:
    if a.answered:
        st.markdown(a.text)
        st.caption(f"Confidence: **{a.confidence_label}** ({a.confidence_pct}%)")
    else:
        st.warning(a.text)
        if a.best_score is not None:
            st.caption(f"Best similarity found: {a.best_score:.3f} (threshold filtering may apply).")
    if a.sources:
        with st.expander(f"📚 Sources ({len(a.sources)})"):
            for i, s in enumerate(a.sources, 1):
                cited = " · ✅ cited" if i in a.cited else ""
                st.markdown(f"**[{i}] {s.doc_name} — page {s.page}** · similarity `{s.score:.3f}`{cited}")
                st.text_area("Excerpt", s.text, height=110, disabled=True,
                             key=f"{key}-ex-{i}", label_visibility="collapsed")


for idx, item in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(item.question)
    with st.chat_message("assistant"):
        render_answer(item, f"h{idx}")

question = st.chat_input("Ask a question about the selected document(s)…", disabled=not selected)
if question is not None:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and generating…"):
                answer = pipeline.ask(question, selected, top_k, threshold, page_range)
            st.session_state.history.append(answer)
            render_answer(answer, f"h{len(st.session_state.history) - 1}")
        except EmptyQueryError:
            st.warning("Please type a question first.")
        except RAGError as exc:
            st.error(str(exc))
