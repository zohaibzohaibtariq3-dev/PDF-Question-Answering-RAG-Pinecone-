"""Central configuration. Secrets are read from environment variables / .env only."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .exceptions import ConfigError

load_dotenv()

FALLBACK_ANSWER = "The answer is not available in the provided document."
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    pinecone_index: str
    pinecone_cloud: str
    pinecone_region: str
    groq_api_key: str
    groq_model: str
    embedding_model: str
    max_pdf_mb: int
    log_dir: Path

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("PINECONE_API_KEY", self.pinecone_api_key),
                ("GROQ_API_KEY", self.groq_api_key),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                f"Missing environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )


def get_settings() -> Settings:
    return Settings(
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index=os.getenv("PINECONE_INDEX_NAME", "pdf-rag-index-e5").strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "multilingual-e5-large"
        ).strip(),
        max_pdf_mb=int(os.getenv("MAX_PDF_MB", "20")),
        log_dir=Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs"))),
    )

