"""Semantic retrieval: embed the query, search selected namespaces, apply threshold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .exceptions import EmptyQueryError, RAGError


class _Embedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class _Store(Protocol):
    def query(
        self, namespace: str, vector: list[float], top_k: int, metadata_filter: dict | None = None
    ) -> list[dict]: ...


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_name: str
    page: int
    text: str
    score: float  # cosine similarity


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]  # chunks that passed the similarity threshold
    best_score: float | None  # best score before thresholding (for diagnostics)


def build_page_filter(page_range: tuple[int, int] | None) -> dict | None:
    if page_range is None:
        return None
    lo, hi = page_range
    if lo > hi:
        lo, hi = hi, lo
    return {"page": {"$gte": lo, "$lte": hi}}


class Retriever:
    def __init__(self, store: _Store, embedder: _Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        namespaces: list[str],
        top_k: int = 4,
        threshold: float = 0.10,
        page_range: tuple[int, int] | None = None,
    ) -> RetrievalResult:
        query = (query or "").strip()
        if not query:
            raise EmptyQueryError("Please type a question first.")
        if not namespaces:
            raise RAGError("Select at least one document to search.")
        if top_k < 1:
            raise RAGError("top_k must be at least 1.")

        vector = self.embedder.embed_query(query)
        metadata_filter = build_page_filter(page_range)

        candidates: list[RetrievedChunk] = []
        for namespace in namespaces:
            for match in self.store.query(namespace, vector, top_k, metadata_filter):
                meta = match["metadata"]
                candidates.append(
                    RetrievedChunk(
                        chunk_id=meta.get("chunk_id", match["id"]),
                        doc_name=meta.get("doc_name", namespace),
                        page=int(meta.get("page", 0)),
                        text=meta.get("text", ""),
                        score=match["score"],
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        top = candidates[:top_k]
        best = top[0].score if top else None
        return RetrievalResult(hits=[c for c in top if c.score >= threshold], best_score=best)
