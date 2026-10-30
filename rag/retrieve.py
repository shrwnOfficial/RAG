"""Hybrid retrieval with RRF and optional cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass

import config
from rag.llm import rewrite_follow_up
from rag.store import DocumentStore, RetrievedChunk


@dataclass
class RetrievalResult:
    standalone_query: str
    chunks: list[RetrievedChunk]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int | None = None,
) -> list[tuple[str, float]]:
    k = k or config.RRF_K
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            from fastembed import TextCrossEncoder  # type: ignore
        _reranker = TextCrossEncoder(model_name=config.RERANKER_MODEL)
    return _reranker


def _rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks or not config.USE_RERANKER:
        return chunks[: config.TOP_K_FINAL]
    model = _get_reranker()
    documents = [c.text for c in chunks]
    scores = list(model.rerank(query, documents))
    paired = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    out: list[RetrievedChunk] = []
    for chunk, score in paired[: config.TOP_K_FINAL]:
        chunk.score = float(score)
        out.append(chunk)
    return out


def retrieve(
    store: DocumentStore,
    question: str,
    chat_history: list[dict] | None = None,
    doc_ids: list[str] | None = None,
) -> RetrievalResult:
    standalone = rewrite_follow_up(question, chat_history or [])
    vector_hits = store.vector_search(standalone, config.TOP_K_VECTOR, doc_ids)
    bm25_hits = store.bm25_search(standalone, config.TOP_K_BM25, doc_ids)

    ranked_lists = [
        [c.chunk_id for c in vector_hits],
        [c.chunk_id for c in bm25_hits],
    ]
    fused = reciprocal_rank_fusion(ranked_lists)
    if not fused:
        return RetrievalResult(standalone_query=standalone, chunks=[])

    candidate_ids = [cid for cid, _ in fused[: max(config.TOP_K_VECTOR, config.TOP_K_BM25)]]
    by_id = {c.chunk_id: c for c in vector_hits + bm25_hits}
    candidates: list[RetrievedChunk] = []
    for cid in candidate_ids:
        if cid in by_id:
            candidates.append(by_id[cid])

    reranked = _rerank(standalone, candidates)
    return RetrievalResult(standalone_query=standalone, chunks=reranked)
