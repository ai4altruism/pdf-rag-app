"""Docling-based PDF parser — converts PDFs to structured Markdown."""

from pathlib import Path

from src.logging import get_logger
from src.pipelines.ingestion.parser import (
    DocumentMetadata,
    DocumentParser,
    PageData,
    ParseResult,
)

logger = get_logger(__name__)


class DoclingParser(DocumentParser):
    """PDF parser using IBM Docling for high-quality structural extraction.

    Converts PDFs to Markdown preserving headers, tables, lists, and reading order.
    Supports OCR for scanned pages via Docling's built-in pipeline.
    """

    @property
    def parser_name(self) -> str:
        return "docling"

    def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF using Docling's DocumentConverter."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        from docling.document_converter import DocumentConverter

        logger.info("docling_parse_start", file=str(file_path))

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = result.document

        # Export to Markdown
        markdown = doc.export_to_markdown()

        # Extract metadata
        meta = DocumentMetadata(
            title=_extract_field(doc, "title"),
            author=_extract_field(doc, "author"),
            language=_extract_field(doc, "language"),
            page_count=doc.num_pages() if hasattr(doc, "num_pages") else 0,
        )

        # Build per-page data from Docling's page mapping
        pages = []
        if hasattr(doc, "pages") and doc.pages:
            for page_no, _page in doc.pages.items():
                page_num = int(page_no) if isinstance(page_no, str) else page_no
                pages.append(PageData(page_number=page_num, text=""))

        if not pages and meta.page_count > 0:
            pages = [PageData(page_number=i + 1) for i in range(meta.page_count)]

        logger.info(
            "docling_parse_complete",
            file=str(file_path),
            pages=meta.page_count,
            markdown_len=len(markdown),
        )

        return ParseResult(markdown=markdown, metadata=meta, pages=pages)


def _extract_field(doc, field_name: str) -> str | None:
    """Safely extract a metadata field from a Docling document."""
    # Docling stores metadata in doc.origin or doc.description depending on version
    for attr in ("origin", "description", "metadata"):
        obj = getattr(doc, attr, None)
        if obj is not None:
            val = getattr(obj, field_name, None)
            if val:
                return str(val)
    return None
