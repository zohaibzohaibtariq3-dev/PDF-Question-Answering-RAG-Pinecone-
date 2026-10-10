"""Sentence-Transformers wrapper. Vectors are L2-normalised so cosine == dot product."""
from __future__ import annotations

import logging

from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"Could not load embedding model: {exc}") from exc
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query])[0]
