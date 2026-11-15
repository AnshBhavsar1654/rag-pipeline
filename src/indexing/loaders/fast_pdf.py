"""
Indexing — fast PDF loader (PyMuPDF)

One pass over the PDF gives each page's text, its tables (as Markdown) and,
when Tesseract is installed, OCR of scanned pages and embedded images.
PyMuPDF parses roughly an order of magnitude faster than pypdf, and replaces
the separate pdfplumber pass that table detection used to need.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document

from src.config import ExtractionConfig
from src.indexing.loaders.table_extractor import table_to_markdown


def available() -> bool:
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        return False
    return True


def _tesseract() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def _ocr_png(png: bytes, language: str) -> str:
    import io

    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(io.BytesIO(png)), lang=language).strip()


def load_pdf(
    path: Path,
    config: ExtractionConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Document]:
    """Load a PDF: one ``text`` document per page, plus ``table`` / OCR documents."""
    import pymupdf

    extract = config is not None and config.enabled
    tables_on = extract and config.table_extraction
    ocr_on = extract and config.ocr_enabled and _tesseract()
    if extract and config.ocr_enabled and not ocr_on:
        print("  [WARN]  Tesseract OCR not found; skipping OCR of scanned pages and images.")

    source = str(path)
    docs: list[Document] = []
    n_tables = n_ocr = 0
    with pymupdf.open(path) as pdf:
        total = pdf.page_count
        for number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": source, "page": number, "type": "text"}))

            if tables_on and _may_have_table(page):
                try:
                    for index, table in enumerate(page.find_tables().tables):
                        rows = table.extract()
                        if len(rows) < config.min_table_rows:
                            continue
                        md = table_to_markdown(rows)
                        if md.strip():
                            n_tables += 1
                            docs.append(Document(page_content=md, metadata={
                                "source": source, "page": number, "type": "table", "table_index": index,
                            }))
                except Exception as e:  # noqa: BLE001
                    print(f"  [WARN]  Table detection failed on page {number}: {e}")

            if ocr_on:
                n_ocr += _ocr_page(page, number, source, text, config, docs)

            if progress and (number % 10 == 0 or number == total):
                progress(number, total)

    print(f"  [FILE] Loaded {total} page(s) from {path.name}"
          + (f", {n_tables} table(s)" if n_tables else "") + (f", {n_ocr} OCR text(s)" if n_ocr else ""))
    return docs


def _may_have_table(page) -> bool:
    """Cheap pre-check: tables are drawn with ruling lines or are dense text grids."""
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return True
    lines = sum(1 for d in drawings for item in d.get("items", []) if item and item[0] in ("l", "re"))
    return lines >= 4


def _ocr_page(page, number: int, source: str, text: str, config: ExtractionConfig, docs: list[Document]) -> int:
    """OCR a scanned page (little text, has images) or its embedded images."""
    added = 0
    images = page.get_images(full=True)
    if not images:
        return 0
    try:
        if config.detect_scanned_pages and len(text) < 50:
            ocr = _ocr_png(page.get_pixmap(dpi=200).tobytes("png"), config.ocr_language)
            if len(ocr) > 50:
                docs.append(Document(page_content=ocr, metadata={
                    "source": source, "page": number, "type": "scanned_page_ocr"}))
                return 1
        for index, image in enumerate(images):
            pix = page.parent.extract_image(image[0])
            if not pix or pix.get("width", 0) < 100 or pix.get("height", 0) < 60:
                continue  # icons and decorations
            ocr = _ocr_png(pix["image"], config.ocr_language) if pix.get("ext") == "png" else ""
            if not ocr:
                import pymupdf
                ocr = _ocr_png(pymupdf.Pixmap(page.parent, image[0]).tobytes("png"), config.ocr_language)
            if len(ocr) > 20:
                docs.append(Document(page_content=ocr, metadata={
                    "source": source, "page": number, "type": "image_ocr", "image_index": index}))
                added += 1
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN]  OCR failed on page {number}: {e}")
    return added
