import io

import pytest
from pypdf import PdfWriter

from rag.exceptions import InvalidPDFError
from rag.pdf_loader import load_pdf


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return out


def test_valid_pdf_is_loaded():
    pdf = load_pdf(_minimal_pdf("Hello RAG world"), "ok.pdf")
    assert pdf.total_pages == 1
    assert "Hello RAG world" in pdf.pages[0].text
    assert len(pdf.doc_id) == 10


def test_empty_file_rejected():
    with pytest.raises(InvalidPDFError):
        load_pdf(b"", "empty.pdf")


def test_non_pdf_rejected():
    with pytest.raises(InvalidPDFError, match="not a valid PDF"):
        load_pdf(b"just some text, not a pdf", "fake.pdf")


def test_corrupt_pdf_rejected():
    with pytest.raises(InvalidPDFError):
        load_pdf(b"%PDF-1.4\ngarbage garbage", "corrupt.pdf")


def test_oversized_pdf_rejected():
    with pytest.raises(InvalidPDFError, match="larger than"):
        load_pdf(b"%PDF-" + b"0" * (2 * 1024 * 1024), "big.pdf", max_mb=1)


def test_pdf_without_text_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(InvalidPDFError, match="no extractable text"):
        load_pdf(buf.getvalue(), "blank.pdf")
