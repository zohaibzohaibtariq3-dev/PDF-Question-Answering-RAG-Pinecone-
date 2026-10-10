"""PDF loading + validation. Returns raw per-page text; cleaning happens elsewhere."""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass

from pypdf import PdfReader

from .exceptions import InvalidPDFError

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    page: int  # 1-based physical page index
    text: str


@dataclass
class LoadedPDF:
    filename: str
    doc_id: str  # short content hash -> stable id, enables de-duplication
    pages: list[PageText]

    @property
    def total_pages(self) -> int:
        return len(self.pages)


def load_pdf(data: bytes, filename: str, max_mb: int = 20) -> LoadedPDF:
    """Validate and extract text from a PDF. Raises InvalidPDFError on any problem."""
    if not data:
        raise InvalidPDFError(f"'{filename}' is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise InvalidPDFError(f"'{filename}' is larger than the {max_mb} MB limit.")
    if data[:1024].find(b"%PDF-") == -1:
        raise InvalidPDFError(f"'{filename}' is not a valid PDF (missing PDF header).")

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:  # noqa: BLE001 - pypdf raises various errors here
                unlocked = 0
            if not unlocked:
                raise InvalidPDFError(f"'{filename}' is password protected.")
        page_objects = list(reader.pages)
    except InvalidPDFError:
        raise
    except Exception as exc:  # noqa: BLE001 - corrupt files raise many error types
        raise InvalidPDFError(f"'{filename}' could not be read as a PDF: {exc}") from exc

    if not page_objects:
        raise InvalidPDFError(f"'{filename}' contains no pages.")

    pages: list[PageText] = []
    for number, page in enumerate(page_objects, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract page %d of %s: %s", number, filename, exc)
            text = ""
        pages.append(PageText(page=number, text=text))

    if not any(p.text.strip() for p in pages):
        raise InvalidPDFError(
            f"'{filename}' has no extractable text (it may be a scanned image PDF)."
        )

    doc_id = hashlib.sha256(data).hexdigest()[:10]
    logger.info("Loaded %s: %d pages, doc_id=%s", filename, len(pages), doc_id)
    return LoadedPDF(filename=filename, doc_id=doc_id, pages=pages)
