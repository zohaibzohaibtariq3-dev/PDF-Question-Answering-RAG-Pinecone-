"""Text cleaning and chunking.

Chunks never cross page boundaries, so every chunk maps to exactly one page number
(needed for traceable source attribution).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_loader import PageText

MIN_CHUNK_CHARS = 15

_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    doc_id: str
    page: int
    chunk_index: int
    text: str


def _signature(line: str) -> str:
    """Normalise a line so 'Page 3' and 'Page 4' (running headers) compare equal."""
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_headers_footers(
    pages: list[PageText], min_pages: int = 4, ratio: float = 0.5, edge_lines: int = 2
) -> list[PageText]:
    """Drop lines that repeat at the top/bottom of at least `ratio` of the pages."""
    if len(pages) < min_pages:
        return pages

    counts: Counter[str] = Counter()
    for p in pages:
        lines = [ln for ln in p.text.splitlines() if ln.strip()]
        n = min(edge_lines, len(lines) // 2)  # top/bottom regions must never cover the whole page
        edges = {_signature(ln) for ln in lines[:n] + lines[len(lines) - n :]}
        counts.update(edges)
    repeated = {sig for sig, c in counts.items() if sig and c / len(pages) >= ratio}
    if not repeated:
        return pages

    cleaned: list[PageText] = []
    for p in pages:
        lines = p.text.splitlines()
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        n = min(edge_lines, len(non_empty) // 2)
        edge_idx = set(non_empty[:n] + non_empty[len(non_empty) - n :])
        kept = [
            ln for i, ln in enumerate(lines) if not (i in edge_idx and _signature(ln) in repeated)
        ]
        cleaned.append(PageText(page=p.page, text="\n".join(kept)))
    return cleaned


def clean_text(text: str) -> str:
    """Remove common PDF extraction artifacts while keeping paragraph structure."""
    text = unicodedata.normalize("NFKC", text)  # ligatures: ﬁ -> fi
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\u00ad", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated at line end
    lines = [ln for ln in text.splitlines() if not _PAGE_NUM_RE.match(ln)]
    paragraphs = re.split(r"\n\s*\n", "\n".join(lines))
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def chunk_pages(
    pages: list[PageText], doc_name: str, doc_id: str, chunk_size: int = 800, overlap: int = 100
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        for idx, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-p{page.page}-c{idx}",
                    doc_name=doc_name,
                    doc_id=doc_id,
                    page=page.page,
                    chunk_index=idx,
                    text=piece,
                )
            )
    return chunks


def prepare_chunks(
    pages: list[PageText], doc_name: str, doc_id: str, chunk_size: int = 800, overlap: int = 100
) -> list[Chunk]:
    """Full text pipeline: strip headers/footers -> clean -> chunk."""
    pages = remove_headers_footers(pages)
    cleaned = [PageText(page=p.page, text=clean_text(p.text)) for p in pages]
    return chunk_pages(cleaned, doc_name, doc_id, chunk_size, overlap)
