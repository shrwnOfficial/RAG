"""PDF text extraction with watermark filtering and OCR fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
from PIL import Image

import config
from rag.ingest.ocr import ocr_image, page_needs_ocr


@dataclass
class PageContent:
    page_num: int
    text: str
    source_type: str  # "text" | "ocr"


def _is_horizontal(span: dict) -> bool:
    direction = span.get("dir", (1, 0))
    if not direction or len(direction) < 2:
        return True
    dx, dy = direction[0], direction[1]
    return abs(dx) >= 0.9 and abs(dy) <= 0.2


def extract_page_text(page: fitz.Page) -> str:
    """Extract text, skipping rotated watermark spans."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
    lines_out: list[str] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            if not all(_is_horizontal(s) for s in spans):
                continue
            line_text = "".join(s.get("text", "") for s in spans).strip()
            if line_text:
                lines_out.append(line_text)
    return "\n".join(lines_out).strip()


def render_page_image(doc: fitz.Document, page_index: int, dpi: int | None = None) -> Image.Image:
    dpi = dpi or config.OCR_DPI
    page = doc[page_index]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def load_pdf(path: Path, on_page=None) -> list[PageContent]:
    """
    Load PDF pages. Calls on_page(current, total) if provided for progress.
    """
    pages: list[PageContent] = []
    doc = fitz.open(path)
    total = doc.page_count
    try:
        for i in range(total):
            if on_page:
                on_page(i + 1, total)
            page = doc[i]
            text = extract_page_text(page)
            if page_needs_ocr(text):
                image = render_page_image(doc, i)
                text = ocr_image(image)
                source = "ocr"
            else:
                source = "text"
            pages.append(PageContent(page_num=i + 1, text=text, source_type=source))
    finally:
        doc.close()
    return pages
