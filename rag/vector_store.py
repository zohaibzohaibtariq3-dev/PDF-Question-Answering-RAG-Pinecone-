"""Pinecone integration: index creation, namespaces, upsert, query, metadata, deletion."""
from __future__ import annotations

import logging
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from .exceptions import (
    ConfigError,
    RAGError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from .text_processing import Chunk

logger = logging.getLogger(__name__)


def _summary_count(summary: Any) -> int:
    if isinstance(summary, dict):
        return int(summary.get("vector_count", 0))
    return int(getattr(summary, "vector_count", 0))


class PineconeStore:
    """Thin wrapper around a serverless Pinecone index (cosine metric).

    One *namespace* per document keeps documents isolated and lets us delete or
    query a single document (or a selection of them) without touching the rest.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        if not api_key:
            raise ConfigError("PINECONE_API_KEY is not set.")
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self._pc = Pinecone(api_key=api_key)
        self._index = None

    # ------------------------------------------------------------------ index
    def connect(self) -> None:
        """Create the index if it does not exist, then open it."""
        try:
            if self.index_name not in self._pc.list_indexes().names():
                logger.info("Creating Pinecone index %s (dim=%d)", self.index_name, self.dimension)
                self._pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
                self._wait_until_ready()
            else:
                desc = self._pc.describe_index(self.index_name)
                if int(desc.dimension) != self.dimension:
                    raise VectorStoreError(
                        f"Index '{self.index_name}' has dimension {desc.dimension} but the "
                        f"embedding model produces {self.dimension}. Use another "
                        "PINECONE_INDEX_NAME or delete the old index."
                    )
            self._index = self._pc.Index(self.index_name)
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreConnectionError(f"Could not connect to Pinecone: {exc}") from exc

    def _wait_until_ready(self, timeout: float = 90.0) -> None:
        start = time.time()
        while time.time() - start < timeout:
            status = self._pc.describe_index(self.index_name).status
            ready = status["ready"] if isinstance(status, dict) else getattr(status, "ready", False)
            if ready:
                return
            time.sleep(2)
        raise VectorStoreConnectionError("Timed out waiting for the Pinecone index to be ready.")

    @property
    def index(self):
        if self._index is None:
            self.connect()
        return self._index

    # ------------------------------------------------------------- namespaces
    def namespace_counts(self) -> dict[str, int]:
        try:
            stats = self.index.describe_index_stats()
            return {ns: _summary_count(s) for ns, s in stats.namespaces.items()}
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreConnectionError(f"Could not read index stats: {exc}") from exc

    def delete_namespace(self, namespace: str) -> None:
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "not found" in msg or "404" in msg:  # namespace does not exist yet
                return
            raise VectorStoreError(f"Could not delete namespace '{namespace}': {exc}") from exc

    # ----------------------------------------------------------------- upsert
    def upsert_chunks(
        self,
        namespace: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        chunk_size: int,
        batch_size: int = 100,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        records = [
            {
                "id": c.chunk_id,
                "values": v,
                "metadata": {
                    "doc_name": c.doc_name,
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "chunk_size": chunk_size,
                    "text": c.text,  # stored so answers can show the exact excerpt
                },
            }
            for c, v in zip(chunks, vectors)
        ]
        try:
            for i in range(0, len(records), batch_size):
                self.index.upsert(vectors=records[i : i + batch_size], namespace=namespace)
            self._wait_for_count(namespace, len(records))
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Upsert failed: {exc}") from exc
        logger.info("Upserted %d vectors into namespace %s", len(records), namespace)
        return len(records)

    def _wait_for_count(self, namespace: str, expected: int, timeout: float = 20.0) -> None:
        """Pinecone is eventually consistent: wait until the new vectors are queryable."""
        start = time.time()
        while time.time() - start < timeout:
            if self.namespace_counts().get(namespace, 0) >= expected:
                return
            time.sleep(1)
        logger.warning("Namespace %s did not reach %d vectors within %.0fs", namespace, expected, timeout)

    # ------------------------------------------------------------------ query
    def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True,
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        try:
            res = self.index.query(**kwargs)
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreConnectionError(f"Pinecone query failed: {exc}") from exc
        return [
            {"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})}
            for m in res.matches
        ]
