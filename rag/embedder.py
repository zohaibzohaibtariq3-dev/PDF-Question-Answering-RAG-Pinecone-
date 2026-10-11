import os
from pinecone import Pinecone
from .exceptions import EmbeddingError


class Embedder:
    def __init__(self, model_name: str = "multilingual-e5-large"):
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise EmbeddingError("PINECONE_API_KEY is missing")
            self._client = Pinecone(api_key=api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return 1024

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        try:
            result = self.client.inference.embed(
                model=self.model_name,
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )
            return [item.values for item in result.data]
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        try:
            result = self.client.inference.embed(
                model=self.model_name,
                inputs=[query],
                parameters={"input_type": "query", "truncate": "END"},
            )
            return result.data[0].values
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc
