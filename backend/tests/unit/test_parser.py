"""Tests for parser abstraction, factory, and parser implementations."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipelines.ingestion import create_parser
from src.pipelines.ingestion.parser import DocumentMetadata, DocumentParser, PageData, ParseResult


@pytest.fixture
def mock_docling():
    """Mock docling package for tests that need it."""
    mock_converter = MagicMock()
    mock_module = MagicMock()
    mock_module.DocumentConverter = mock_converter

    with patch.dict(sys.modules, {
        "docling": MagicMock(),
        "docling.document_converter": mock_module,
    }):
        yield mock_converter


@pytest.fixture
def mock_pymupdf_libs():
    """Mock pymupdf and pymupdf4llm packages for tests."""
    mock_pymupdf = MagicMock()
    mock_pymupdf4llm = MagicMock()

    with patch.dict(sys.modules, {
        "pymupdf": mock_pymupdf,
        "pymupdf4llm": mock_pymupdf4llm,
    }):
        yield mock_pymupdf, mock_pymupdf4llm


# --- Parser ABC tests ---


class TestDocumentParserABC:
    """Tests for the DocumentParser abstract interface."""

    def test_cannot_instantiate_abc(self):
        """DocumentParser cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DocumentParser()

    def test_concrete_implementation_works(self):
        """A concrete subclass can be instantiated."""

        class StubParser(DocumentParser):
            @property
            def parser_name(self) -> str:
                return "stub"

            def parse(self, file_path: Path) -> ParseResult:
                return ParseResult(markdown="# Test")

        parser = StubParser()
        assert parser.parser_name == "stub"
        result = parser.parse(Path("test.pdf"))
        assert result.markdown == "# Test"


# --- ParseResult / data structure tests ---


class TestParseResult:
    """Tests for ParseResult and its component dataclasses."""

    def test_default_values(self):
        result = ParseResult(markdown="hello")
        assert result.markdown == "hello"
        assert result.metadata.title is None
        assert result.metadata.page_count == 0
        assert result.pages == []

    def test_with_metadata(self):
        meta = DocumentMetadata(title="My Doc", author="Alice", language="en", page_count=5)
        pages = [PageData(page_number=i, text=f"page {i}") for i in range(1, 6)]
        result = ParseResult(markdown="# My Doc", metadata=meta, pages=pages)
        assert result.metadata.title == "My Doc"
        assert len(result.pages) == 5
        assert result.pages[0].page_number == 1


# --- Factory tests ---


class TestCreateParser:
    """Tests for the parser factory function."""

    def test_create_docling_parser(self):
        parser = create_parser("docling")
        assert parser.parser_name == "docling"

    def test_create_pymupdf4llm_parser(self):
        parser = create_parser("pymupdf4llm")
        assert parser.parser_name == "pymupdf4llm"

    def test_create_parser_with_enum(self):
        from src.config import ParserType

        parser = create_parser(ParserType.DOCLING)
        assert parser.parser_name == "docling"

    def test_vision_parser_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Sprint 8"):
            create_parser("vision")

    def test_unknown_parser_raises(self):
        with pytest.raises(ValueError):
            create_parser("nonexistent")


# --- DoclingParser tests ---


class TestDoclingParser:
    """Tests for the DoclingParser with mocked Docling library."""

    def test_parse_file_not_found(self):
        parser = create_parser("docling")
        with pytest.raises(FileNotFoundError):
            parser.parse(Path("/nonexistent/file.pdf"))

    def test_parse_not_pdf(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        parser = create_parser("docling")
        with pytest.raises(ValueError, match="Not a PDF"):
            parser.parse(txt_file)

    def test_parse_success(self, mock_docling, tmp_path):
        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        # Mock Docling's DocumentConverter
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Title\n\nSome content"
        mock_doc.num_pages.return_value = 3
        mock_doc.pages = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}

        # Mock metadata extraction
        mock_doc.origin = MagicMock()
        mock_doc.origin.title = "Test Title"
        mock_doc.origin.author = "Test Author"
        mock_doc.origin.language = None

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_docling.return_value.convert.return_value = mock_result

        parser = create_parser("docling")
        result = parser.parse(pdf_file)

        assert "# Title" in result.markdown
        assert result.metadata.title == "Test Title"
        assert result.metadata.author == "Test Author"
        assert result.metadata.page_count == 3
        assert len(result.pages) == 3

    def test_parse_no_pages_attr(self, mock_docling, tmp_path):
        """Parser handles docs without pages attribute gracefully."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "Content"
        mock_doc.num_pages.return_value = 2
        mock_doc.pages = None
        mock_doc.origin = None
        mock_doc.description = None
        mock_doc.metadata = None

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_docling.return_value.convert.return_value = mock_result

        parser = create_parser("docling")
        result = parser.parse(pdf_file)

        assert result.markdown == "Content"
        assert result.metadata.page_count == 2
        assert len(result.pages) == 2


# --- PyMuPDF4LLMParser tests ---


class TestPyMuPDF4LLMParser:
    """Tests for the PyMuPDF4LLMParser with mocked pymupdf4llm library."""

    def test_parse_file_not_found(self):
        parser = create_parser("pymupdf4llm")
        with pytest.raises(FileNotFoundError):
            parser.parse(Path("/nonexistent/file.pdf"))

    def test_parse_not_pdf(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        parser = create_parser("pymupdf4llm")
        with pytest.raises(ValueError, match="Not a PDF"):
            parser.parse(txt_file)

    def test_parse_success(self, mock_pymupdf_libs, tmp_path):
        mock_pymupdf, mock_pymupdf4llm = mock_pymupdf_libs
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        # Mock pymupdf4llm.to_markdown
        mock_pymupdf4llm.to_markdown.return_value = "# Heading\n\nParagraph text"

        # Mock pymupdf.open
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 text"
        mock_pdf = MagicMock()
        mock_pdf.__len__ = MagicMock(return_value=2)
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)
        mock_pdf.metadata = {"title": "My Title", "author": "Bob"}
        mock_pymupdf.open.return_value = mock_pdf

        # Need to reimport to pick up mocked modules
        import importlib

        import src.pipelines.ingestion.pymupdf4llm_parser as mod
        importlib.reload(mod)

        parser = mod.PyMuPDF4LLMParser()
        result = parser.parse(pdf_file)

        assert "# Heading" in result.markdown
        assert result.metadata.title == "My Title"
        assert result.metadata.author == "Bob"
        assert result.metadata.page_count == 2
        assert len(result.pages) == 2

    def test_parse_empty_metadata(self, mock_pymupdf_libs, tmp_path):
        """Parser handles empty/missing metadata gracefully."""
        mock_pymupdf, mock_pymupdf4llm = mock_pymupdf_libs
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        mock_pymupdf4llm.to_markdown.return_value = "Some text"
        mock_pdf = MagicMock()
        mock_pdf.__len__ = MagicMock(return_value=1)
        mock_pdf.__getitem__ = MagicMock(return_value=MagicMock(get_text=MagicMock(return_value="")))
        mock_pdf.metadata = {"title": "", "author": ""}
        mock_pymupdf.open.return_value = mock_pdf

        import importlib

        import src.pipelines.ingestion.pymupdf4llm_parser as mod
        importlib.reload(mod)

        parser = mod.PyMuPDF4LLMParser()
        result = parser.parse(pdf_file)

        assert result.metadata.title is None
        assert result.metadata.author is None
