import pytest

from rag.pdf_loader import PageText
from rag.text_processing import (
    chunk_pages,
    clean_text,
    prepare_chunks,
    remove_headers_footers,
)


def test_clean_text_fixes_hyphenation_and_ligatures():
    raw = "The inter-\nnational ﬁnancial report"
    assert clean_text(raw) == "The international financial report"


def test_clean_text_removes_page_numbers_and_collapses_whitespace():
    raw = "Intro   text\nspread over\nlines\n\n12\n\nPage 3 of 10\n\nNext   paragraph"
    assert clean_text(raw) == "Intro text spread over lines\n\nNext paragraph"


def test_remove_repeated_headers_and_footers():
    pages = [
        PageText(i, f"ACME Annual Report\nContent of page {i} is unique.\nPage {i}")
        for i in range(1, 7)
    ]
    out = remove_headers_footers(pages)
    assert all("ACME Annual Report" not in p.text for p in out)
    assert all("Content of page" in p.text for p in out)


def test_headers_kept_for_short_documents():
    pages = [PageText(1, "Title\nBody one"), PageText(2, "Title\nBody two")]
    assert remove_headers_footers(pages) == pages


def test_chunks_carry_metadata_and_stay_on_one_page():
    pages = [PageText(1, "Alpha beta gamma. " * 60), PageText(2, "Second page text here. " * 40)]
    chunks = chunk_pages(pages, "doc.pdf", "abc123", chunk_size=200, overlap=20)
    assert len(chunks) > 2
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].chunk_id == "abc123-p1-c0"
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # unique ids


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_settings_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "x" * 50)], "d", "i", size, overlap)


def test_prepare_chunks_end_to_end_skips_empty_pages():
    pages = [PageText(1, "Hello world, this is real content."), PageText(2, "   \n  ")]
    chunks = prepare_chunks(pages, "d.pdf", "id1", 500, 50)
    assert len(chunks) == 1 and chunks[0].page == 1
