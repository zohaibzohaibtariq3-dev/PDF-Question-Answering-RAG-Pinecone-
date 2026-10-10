"""Project-specific exceptions so the UI can show friendly, precise errors."""


class RAGError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigError(RAGError):
    """Missing or invalid configuration (e.g. API key not set)."""


class InvalidPDFError(RAGError):
    """The uploaded file is not a readable PDF (corrupt, empty, scanned, too large...)."""


class EmptyQueryError(RAGError):
    """The user submitted an empty question."""


class EmbeddingError(RAGError):
    """The embedding model could not be loaded or run."""


class VectorStoreError(RAGError):
    """A Pinecone operation failed."""


class VectorStoreConnectionError(VectorStoreError):
    """Could not reach / authenticate with Pinecone."""


class GenerationError(RAGError):
    """The LLM call failed."""
