"""Document parser abstraction — ABC and data structures for PDF parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageData:
    """Data extracted from a single PDF page."""

    page_number: int
    text: str = ""


@dataclass
class DocumentMetadata:
    """Metadata extracted from a parsed document."""

    title: str | None = None
    author: str | None = None
    language: str | None = None
    page_count: int = 0


@dataclass
class ParseResult:
    """Complete result of parsing a PDF document."""

    markdown: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    pages: list[PageData] = field(default_factory=list)


class DocumentParser(ABC):
    """Abstract interface for PDF-to-Markdown parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParseResult:
        """Parse a PDF file and return structured Markdown with metadata.

        Args:
            file_path: Path to the PDF file on disk.

        Returns:
            ParseResult with markdown text, document metadata, and per-page data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        ...

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """Identifier string for this parser (e.g. 'docling', 'pymupdf4llm')."""
        ...
