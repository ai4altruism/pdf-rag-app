"""PyMuPDF4LLM-based PDF parser — fallback Markdown conversion."""

from pathlib import Path

from src.logging import get_logger
from src.pipelines.ingestion.parser import (
    DocumentMetadata,
    DocumentParser,
    PageData,
    ParseResult,
)

logger = get_logger(__name__)


class PyMuPDF4LLMParser(DocumentParser):
    """PDF parser using PyMuPDF4LLM for fast Markdown conversion.

    Lighter-weight alternative to Docling. Good for well-structured,
    text-heavy PDFs. Less capable with complex tables or scanned pages.
    """

    @property
    def parser_name(self) -> str:
        return "pymupdf4llm"

    def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF using PyMuPDF4LLM."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        import pymupdf
        import pymupdf4llm

        logger.info("pymupdf4llm_parse_start", file=str(file_path))

        # Get structured Markdown with page breaks
        markdown = pymupdf4llm.to_markdown(str(file_path))

        # Extract metadata from PyMuPDF
        pdf_doc = pymupdf.open(str(file_path))
        pdf_metadata = pdf_doc.metadata or {}
        page_count = len(pdf_doc)

        meta = DocumentMetadata(
            title=pdf_metadata.get("title") or None,
            author=pdf_metadata.get("author") or None,
            language=None,  # PyMuPDF doesn't reliably detect language
            page_count=page_count,
        )

        # Build per-page data
        pages = []
        for i in range(page_count):
            page = pdf_doc[i]
            pages.append(PageData(page_number=i + 1, text=page.get_text()))

        pdf_doc.close()

        logger.info(
            "pymupdf4llm_parse_complete",
            file=str(file_path),
            pages=page_count,
            markdown_len=len(markdown),
        )

        return ParseResult(markdown=markdown, metadata=meta, pages=pages)
