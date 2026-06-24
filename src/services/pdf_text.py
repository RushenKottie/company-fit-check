"""Local PDF text extraction."""

from io import BytesIO

from pypdf import PdfReader

from logging_utils import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes locally."""

    if not pdf_bytes:
        raise ValueError("CV PDF is empty.")

    logger.info("Extracting text from PDF bytes=%s", len(pdf_bytes))
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page.strip() for page in pages if page.strip()).strip()
    if not text:
        raise ValueError("No text could be extracted from the CV PDF.")
    logger.info("Extracted PDF text pages=%s chars=%s", len(reader.pages), len(text))
    return text
