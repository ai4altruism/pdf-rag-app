"""Ingestion pipeline — PDF parsing, chunking, and embedding."""

from src.config import ParserType
from src.pipelines.ingestion.parser import DocumentParser


def create_parser(parser_type: str | ParserType) -> DocumentParser:
    """Factory function to create a parser from config.

    Args:
        parser_type: One of 'docling', 'pymupdf4llm', or 'vision'.

    Returns:
        A DocumentParser instance.

    Raises:
        ValueError: If the parser type is unknown.
    """
    parser_type = ParserType(parser_type)

    match parser_type:
        case ParserType.DOCLING:
            from src.pipelines.ingestion.docling_parser import DoclingParser

            return DoclingParser()
        case ParserType.PYMUPDF4LLM:
            from src.pipelines.ingestion.pymupdf4llm_parser import PyMuPDF4LLMParser

            return PyMuPDF4LLMParser()
        case ParserType.VISION:
            raise NotImplementedError("Vision parser is planned for Sprint 8")
        case _:
            raise ValueError(f"Unknown parser type: {parser_type}")
