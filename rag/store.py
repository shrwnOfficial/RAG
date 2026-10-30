"""Chroma vector store and BM25 keyword index."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

import config
from rag.ingest.chunker import ChunkRecord


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    section: str
    text: str
    source_type: str
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class DocumentStore:
    def __init__(self) -> None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = TextEmbedding(model_name=config.EMBEDDING_MODEL)
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: list[str] = []
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        result = self._collection.get(include=["documents"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        self._bm25_ids = list(ids)
        self._bm25_docs = list(docs)
        if docs:
            tokenized = [_tokenize(d) for d in docs]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return list(self._embedder.embed(texts))

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "filename": c.filename,
                "page": c.page,
                "section": c.section or "",
                "source_type": c.source_type,
                "company": c.company or "",
                "ref_no": c.ref_no or "",
            }
            for c in chunks
        ]
        embeddings = self._embed(documents)
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._rebuild_bm25()

    def delete_by_doc_id(self, doc_id: str) -> None:
        existing = self._collection.get(where={"doc_id": doc_id})
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        self._rebuild_bm25()

    def clear(self) -> None:
        name = config.CHROMA_COLLECTION
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self._rebuild_bm25()

    def chunk_count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[dict]:
        result = self._collection.get(include=["metadatas"])
        metas = result.get("metadatas") or []
        by_doc: dict[str, dict] = {}
        for meta in metas:
            if not meta:
                continue
            doc_id = meta["doc_id"]
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", ""),
                    "chunk_count": 0,
                }
            by_doc[doc_id]["chunk_count"] += 1
        return sorted(by_doc.values(), key=lambda x: x["filename"].lower())

    def _row_to_chunk(self, chunk_id: str, doc: str, meta: dict, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=meta.get("doc_id", ""),
            filename=meta.get("filename", ""),
            page=int(meta.get("page", 0)),
            section=meta.get("section", ""),
            text=doc,
            source_type=meta.get("source_type", "text"),
            score=score,
        )

    def vector_search(
        self,
        query: str,
        top_k: int,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []
        where = None
        if doc_ids:
            if len(doc_ids) == 1:
                where = {"doc_id": doc_ids[0]}
            else:
                where = {"doc_id": {"$in": doc_ids}}
        q_emb = self._embed([query])[0]
        result = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist) if dist is not None else 0.0
            out.append(self._row_to_chunk(cid, doc, meta or {}, score))
        return out

    def bm25_search(
        self,
        query: str,
        top_k: int,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._bm25 or not self._bm25_ids:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._bm25_ids, self._bm25_docs, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        out: list[RetrievedChunk] = []
        for chunk_id, doc, score in ranked:
            if score <= 0:
                continue
            meta_result = self._collection.get(ids=[chunk_id], include=["metadatas"])
            metas = meta_result.get("metadatas") or []
            if not metas:
                continue
            meta = metas[0] or {}
            if doc_ids and meta.get("doc_id") not in doc_ids:
                continue
            out.append(self._row_to_chunk(chunk_id, doc, meta, float(score)))
            if len(out) >= top_k:
                break
        return out

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return [
            self._row_to_chunk(cid, doc, meta or {}, 0.0)
            for cid, doc, meta in zip(ids, docs, metas)
        ]
