from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


def _make_overlay(width: float, height: float) -> BytesIO:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))
    c.setFillColorRGB(1, 1, 1)
    # Cover top third
    c.rect(0, height * (2.0 / 3.0), width, height / 3.0, fill=1, stroke=0)
    c.save()
    buf.seek(0)
    return buf


def redact_pdf_top_third(in_pdf: Path, out_pdf: Path) -> bool:
    """Overlay a white rectangle over top third of every page and save as out_pdf.

    Returns True on success.
    """
    try:
        reader = PdfReader(str(in_pdf))
        writer = PdfWriter()
        for page in reader.pages:
            mediabox = page.mediabox
            width = float(mediabox.width)
            height = float(mediabox.height)
            overlay_buf = _make_overlay(width, height)
            overlay_reader = PdfReader(overlay_buf)
            overlay_page = overlay_reader.pages[0]
            page.merge_page(overlay_page)
            writer.add_page(page)

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        with out_pdf.open("wb") as f:
            writer.write(f)
        return True
    except Exception:
        return False


def extract_study_text_segment(in_pdf: Path) -> str:
    """Extract text from first occurrence of 'Study' up to before 'Electronically Signed by'.

    Returns empty string if not found.
    """
    try:
        reader = PdfReader(str(in_pdf))
        text_parts: list[str] = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        full_text = "\n".join(text_parts)
        start_idx = full_text.find("Study")
        if start_idx == -1:
            return ""
        end_marker = "Electronically Signed by"
        end_idx = full_text.find(end_marker, start_idx)
        if end_idx == -1:
            # until end if marker not found
            end_idx = len(full_text)
        segment = full_text[start_idx:end_idx].strip()
        return segment
    except Exception:
        return ""


