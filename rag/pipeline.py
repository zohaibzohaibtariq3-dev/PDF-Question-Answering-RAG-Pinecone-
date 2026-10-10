"""Orchestrates the whole RAG flow. The UI only talks to this class."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .embedder import Embedder
from .generator import AnswerGenerator
from .pdf_loader import load_pdf
from .query_logger import log_query
from .retriever import RetrievedChunk, Retriever
from .text_processing import prepare_chunks
from .vector_store import PineconeStore
from .exceptions import InvalidPDFError

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    namespace: str
    doc_name: str
    pages: int
    chunks: int
    chunk_size: int
    overlap: int
    seconds: float


@dataclass
class Answer:
    question: str
    text: str
    answered: bool
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    confidence_pct: float | None = None
    confidence_label: str = "N/A"
    best_score: float | None = None


def make_namespace(filename: str, doc_id: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")[:40] or "doc"
    return f"{stem}-{doc_id}"


def compute_confidence(hits: list[RetrievedChunk]) -> tuple[float | None, str]:
    """Heuristic: mean cosine similarity of the top-3 retrieved chunks."""
    if not hits:
        return None, "N/A"
    top = [h.score for h in hits[:3]]
    mean = sum(top) / len(top)
    label = "High" if mean >= 0.55 else "Medium" if mean >= 0.40 else "Low"
    return round(max(0.0, min(mean, 1.0)) * 100, 1), label


class RAGPipeline:
    def __init__(self, settings: Settings):
        settings.validate()
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model)
        self.store = PineconeStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index,
            dimension=self.embedder.dimension,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
        self.store.connect()
        self.retriever = Retriever(self.store, self.embedder)
        self.generator = AnswerGenerator(settings.groq_api_key, settings.groq_model)

    # --------------------------------------------------------------- ingestion
    def ingest(self, data: bytes, filename: str, chunk_size: int = 800, overlap: int = 100) -> IngestResult:
        start = time.time()
        pdf = load_pdf(data, filename, self.settings.max_pdf_mb)
        chunks = prepare_chunks(pdf.pages, filename, pdf.doc_id, chunk_size, overlap)
        if not chunks:
            raise InvalidPDFError(f"No usable text chunks could be created from '{filename}'.")

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        namespace = make_namespace(filename, pdf.doc_id)
        self.store.delete_namespace(namespace)  # re-indexing with new settings replaces old vectors
        self.store.upsert_chunks(namespace, chunks, vectors, chunk_size)
        return IngestResult(
            namespace=namespace,
            doc_name=filename,
            pages=pdf.total_pages,
            chunks=len(chunks),
            chunk_size=chunk_size,
            overlap=overlap,
            seconds=round(time.time() - start, 2),
        )

    def list_documents(self) -> dict[str, int]:
        return {ns: n for ns, n in self.store.namespace_counts().items() if ns}

    def delete_document(self, namespace: str) -> None:
        self.store.delete_namespace(namespace)

    # ---------------------------------------------------------------- question
    def ask(
        self,
        question: str,
        namespaces: list[str],
        top_k: int = 4,
        threshold: float = 0.30,
        page_range: tuple[int, int] | None = None,
    ) -> Answer:
        result = self.retriever.retrieve(question, namespaces, top_k, threshold, page_range)
        generated = self.generator.generate(question.strip(), result.hits)
        pct, label = compute_confidence(result.hits) if generated.answered else (None, "N/A")
        answer = Answer(
            question=question.strip(),
            text=generated.text,
            answered=generated.answered,
            sources=result.hits,
            cited=generated.cited,
            confidence_pct=pct,
            confidence_label=label,
            best_score=result.best_score,
        )
        log_query(
            self.settings.log_dir,
            {
                "question": answer.question,
                "namespaces": namespaces,
                "top_k": top_k,
                "threshold": threshold,
                "page_range": page_range,
                "answered": answer.answered,
                "best_score": answer.best_score,
                "confidence_pct": answer.confidence_pct,
                "sources": [{"doc": s.doc_name, "page": s.page, "score": round(s.score, 4)} for s in answer.sources],
            },
        )
        return answer
