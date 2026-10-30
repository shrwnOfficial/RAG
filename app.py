"""Streamlit RAG application."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import config
from rag.ingest.loader import (
    clear_all,
    delete_document,
    ingest_file,
    list_indexed_documents,
)
from rag.llm import build_rag_prompt, generate_answer
from rag.retrieve import retrieve
from rag.store import DocumentStore

load_dotenv()

st.set_page_config(page_title="Document RAG Chat", layout="wide")


@st.cache_resource
def get_store() -> DocumentStore:
    return DocumentStore()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "doc_filter" not in st.session_state:
        st.session_state.doc_filter = []


def main() -> None:
    init_session_state()
    store = get_store()

    st.title("Document RAG Chat")
    st.caption("Upload PDFs, Excel, or images — then ask questions grounded in your documents.")

    api_ok = bool(os.getenv("GEMINI_API_KEY", "").strip())
    if not api_ok:
        st.error(
            "Set **GEMINI_API_KEY** in a `.env` file (see `.env.example`) or in your environment."
        )

    docs = list_indexed_documents()
    filter_options = {d["filename"]: d["doc_id"] for d in docs}
    doc_ids_filter: list[str] | None = None

    with st.sidebar:
        st.header("Documents")
        uploaded = st.file_uploader(
            "Upload files",
            type=[ext.lstrip(".") for ext in sorted(config.SUPPORTED_EXTENSIONS)],
            accept_multiple_files=True,
        )
        max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024

        if uploaded:
            for file in uploaded:
                if file.size > max_bytes:
                    st.warning(f"{file.name} exceeds {config.MAX_UPLOAD_MB} MB — skipped.")
                    continue
                progress = st.progress(0.0, text=f"Indexing {file.name}…")
                status = st.empty()

                def on_progress(msg: str, frac: float, _progress=progress, _status=status) -> None:
                    _progress.progress(min(frac, 1.0), text=msg)
                    _status.caption(msg)

                try:
                    suffix = Path(file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = Path(tmp.name)
                    result = ingest_file(tmp_path, store, on_progress=on_progress)
                    tmp_path.unlink(missing_ok=True)
                    docs = list_indexed_documents()
                    filter_options = {d["filename"]: d["doc_id"] for d in docs}
                    if result.skipped:
                        st.info(f"{result.filename} already indexed — skipped.")
                    else:
                        st.success(
                            f"Indexed **{result.filename}** "
                            f"({result.page_count} pages, {result.chunk_count} chunks)."
                        )
                except Exception as exc:
                    st.error(f"Failed to index {file.name}: {exc}")
                finally:
                    progress.empty()
                    status.empty()

        st.divider()

        if docs:
            st.subheader("Indexed")
            selected_names = st.multiselect(
                "Limit search to documents (optional)",
                options=sorted(filter_options.keys()),
                default=[n for n in st.session_state.doc_filter if n in filter_options],
            )
            st.session_state.doc_filter = selected_names
            if selected_names:
                doc_ids_filter = [filter_options[n] for n in selected_names]

            chroma_docs = {d["doc_id"]: d for d in store.list_documents()}
            for meta in sorted(docs, key=lambda x: x["filename"].lower()):
                doc_id = meta["doc_id"]
                chunks = chroma_docs.get(doc_id, {}).get("chunk_count", meta.get("chunk_count", 0))
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"{meta['filename']} ({chunks} chunks)")
                with col2:
                    if st.button("Del", key=f"del_{doc_id}"):
                        delete_document(doc_id, store)
                        get_store.clear()
                        st.rerun()

            if st.button("Clear all documents", type="secondary"):
                clear_all(store)
                st.session_state.messages = []
                get_store.clear()
                st.rerun()
        else:
            st.info("No documents indexed yet. Upload files above.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**[{src['index']}]** `{src['filename']}` p.{src['page']} "
                            f"({src['source_type']})"
                        )
                        st.text(src["preview"])

    if prompt := st.chat_input("Ask about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_ok:
                reply = "Configure GEMINI_API_KEY to chat."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            elif store.chunk_count() == 0:
                reply = "Upload and index documents before asking questions."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    retrieval = retrieve(store, prompt, history, doc_ids_filter)
                    rag_prompt = build_rag_prompt(prompt, retrieval.chunks)
                    answer = st.write_stream(generate_answer(rag_prompt))
                    sources = []
                    for i, chunk in enumerate(retrieval.chunks, start=1):
                        preview = chunk.text[:500] + ("…" if len(chunk.text) > 500 else "")
                        sources.append(
                            {
                                "index": i,
                                "filename": chunk.filename,
                                "page": chunk.page,
                                "source_type": chunk.source_type,
                                "preview": preview,
                            }
                        )
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(
                                    f"**[{src['index']}]** `{src['filename']}` p.{src['page']} "
                                    f"({src['source_type']})"
                                )
                                st.text(src["preview"])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    err = f"Error: {exc}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
