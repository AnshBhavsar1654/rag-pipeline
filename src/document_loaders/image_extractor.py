"""
RAG Pipeline — Image/OCR Extractor

Extracts images from PDFs and runs OCR to recover text content.
Handles both embedded images and scanned pages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.documents import Document

from src.config import ExtractionConfig


def _check_tesseract() -> bool:
    """Check if Tesseract OCR is installed and accessible."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _is_scanned_page(page) -> bool:
    """Detect if a PDF page is likely scanned (no extractable text).

    A page is considered "scanned" if it contains text characters
    but very little extractable content — typically indicating an
    image-based page.
    """
    text = page.extract_text() or ""
    # If very little text but the page has images, likely scanned
    has_images = len(page.images) > 0
    has_little_text = len(text.strip()) < 50
    return has_images and has_little_text


def ocr_image(image_bytes: bytes, language: str = "eng") -> str:
    """Run OCR on raw image bytes.

    Args:
        image_bytes: Raw image data.
        language: Tesseract language code.

    Returns:
        Extracted text string.
    """
    from PIL import Image
    import pytesseract
    import io

    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang=language)
        return text.strip()
    except Exception as e:
        print(f"  [WARN]  OCR failed: {e}")
        return ""


def _ocr_page_image(page, language: str = "eng") -> str:
    """Convert a PDF page to image and run OCR."""
    from PIL import Image
    import pytesseract
    import io

    try:
        # Convert page to image (72 DPI for speed, increase for accuracy)
        page_image = page.to_image(resolution=300)
        img_bytes = io.BytesIO()
        page_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        image = Image.open(img_bytes)
        text = pytesseract.image_to_string(image, lang=language)
        return text.strip()
    except Exception as e:
        print(f"  [WARN]  Page OCR failed: {e}")
        return ""


def extract_images_from_pdf(
    pdf_path: str | Path,
    config: ExtractionConfig,
) -> list[Document]:
    """Extract images from a PDF and run OCR on them.

    For each page:
    - Extracts embedded images and OCRs them
    - For scanned pages (detected via minimal text + images),
      OCRs the entire page

    Args:
        pdf_path: Path to the PDF file.
        config: Extraction configuration.

    Returns:
        List of Document objects with OCR text content.
    """
    import pdfplumber

    pdf_path = Path(pdf_path)
    documents: list[Document] = []

    if not _check_tesseract():
        print("  [WARN]  Tesseract OCR not found. Skipping image extraction.")
        print("  [WARN]  Install: https://github.com/tesseract-ocr/tesseract")
        return documents

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # --- 1. OCR embedded images ---
                for img_idx, img_info in enumerate(page.images or []):
                    try:
                        # Extract image crop from page
                        x0 = img_info.get("x0", 0)
                        y0 = img_info.get("top", 0)
                        x1 = img_info.get("x1", page.width)
                        y1 = img_info.get("bottom", page.height)

                        # Crop and convert to image
                        page_image = page.crop((x0, y0, x1, y1)).to_image(resolution=300)

                        import io
                        img_bytes = io.BytesIO()
                        page_image.save(img_bytes, format="PNG")
                        img_bytes.seek(0)

                        from PIL import Image
                        import pytesseract

                        image = Image.open(img_bytes)
                        text = pytesseract.image_to_string(
                            image, lang=config.ocr_language
                        )
                        text = text.strip()

                        if text and len(text) > 20:  # Only keep meaningful OCR results
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": str(pdf_path),
                                    "type": "image_ocr",
                                    "page": page_num,
                                    "image_index": img_idx,
                                },
                            )
                            documents.append(doc)
                    except Exception as e:
                        print(f"  [WARN]  Image OCR failed on page {page_num}: {e}")

                # --- 2. Full-page OCR for scanned pages ---
                if config.detect_scanned_pages and _is_scanned_page(page):
                    try:
                        text = _ocr_page_image(page, config.ocr_language)
                        if text and len(text) > 50:
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": str(pdf_path),
                                    "type": "scanned_page_ocr",
                                    "page": page_num,
                                },
                            )
                            documents.append(doc)
                    except Exception as e:
                        print(f"  [WARN]  Scanned page OCR failed on page {page_num}: {e}")

    except Exception as e:
        print(f"  [ERROR] Error extracting images from {pdf_path.name}: {e}")

    return documents
