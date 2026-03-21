"""Tests for the ingestion pipeline orchestrator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.models.document import Document
from src.models.ingestion_job import IngestionJob
from src.pipelines.ingestion.parser import DocumentMetadata, PageData, ParseResult
from src.pipelines.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def pipeline_settings() -> Settings:
    return Settings(
        parser="docling",
        chunk_size_tokens=512,
        chunk_overlap_tokens=50,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=1024,
    )


@pytest.fixture
def mock_qdrant_for_pipeline() -> AsyncMock:
    mock = AsyncMock()
    mock.collection_exists.return_value = True
    mock.create_collection.return_value = None
    mock.delete_points_by_document.return_value = None
    mock.upsert_points.return_value = None
    return mock


@pytest.fixture
def mock_embedding() -> AsyncMock:
    mock = AsyncMock()
    mock.embed_texts.return_value = [[0.1] * 1024, [0.2] * 1024]
    mock.model_name = "text-embedding-3-large"
    mock.dimensions = 1024
    return mock


def _make_parse_result() -> ParseResult:
    return ParseResult(
        markdown="# Introduction\n\nFirst paragraph.\n\n## Details\n\nSecond paragraph.",
        metadata=DocumentMetadata(
            title="Test Doc", author="Author", language="en", page_count=2
        ),
        pages=[PageData(page_number=1), PageData(page_number=2)],
    )


class TestIngestionPipeline:
    """Tests for the IngestionPipeline orchestrator."""

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_full_pipeline_success(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_for_pipeline: AsyncMock,
        mock_embedding: AsyncMock,
        pipeline_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline runs all stages and marks document complete."""
        # Set up mock parser
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        # Create a document in DB
        doc = Document(
            collection_id="col-1",
            filename="test.pdf",
            status="pending",
            page_count=2,
        )
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        # Create dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        # Run pipeline
        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_for_pipeline,
            embedding_provider=mock_embedding,
            settings=pipeline_settings,
        )
        await pipeline.ingest(doc.id, pdf_file)

        # Verify document status
        await test_db.refresh(doc)
        assert doc.status == "complete"
        assert doc.title == "Test Doc"
        assert doc.author == "Author"
        assert doc.parser_used == "docling"
        assert doc.chunk_count is not None
        assert doc.chunk_count > 0
        assert doc.embedding_model == "text-embedding-3-large"
        assert doc.error_message is None

        # Verify ingestion job created and completed
        result = await test_db.execute(
            select(IngestionJob).where(IngestionJob.document_id == doc.id)
        )
        job = result.scalar_one()
        assert job.status == "complete"
        assert job.parser == "docling"
        assert job.completed_at is not None

        # Verify Qdrant operations
        mock_qdrant_for_pipeline.delete_points_by_document.assert_called_once()
        mock_qdrant_for_pipeline.upsert_points.assert_called_once()

        # Verify embedding was called
        mock_embedding.embed_texts.assert_called_once()

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_pipeline_parser_failure(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_for_pipeline: AsyncMock,
        mock_embedding: AsyncMock,
        pipeline_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline marks document as failed when parsing raises."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.side_effect = RuntimeError("Parse error")
        mock_create_parser.return_value = mock_parser

        doc = Document(
            collection_id="col-1",
            filename="test.pdf",
            status="pending",
        )
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_for_pipeline,
            embedding_provider=mock_embedding,
            settings=pipeline_settings,
        )
        await pipeline.ingest(doc.id, pdf_file)

        await test_db.refresh(doc)
        assert doc.status == "failed"
        assert "Parse error" in doc.error_message

        result = await test_db.execute(
            select(IngestionJob).where(IngestionJob.document_id == doc.id)
        )
        job = result.scalar_one()
        assert job.status == "failed"

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_pipeline_embedding_failure(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_for_pipeline: AsyncMock,
        mock_embedding: AsyncMock,
        pipeline_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline marks document as failed when embedding raises."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        mock_embedding.embed_texts.side_effect = RuntimeError("API error")

        doc = Document(
            collection_id="col-1",
            filename="test.pdf",
            status="pending",
        )
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_for_pipeline,
            embedding_provider=mock_embedding,
            settings=pipeline_settings,
        )
        await pipeline.ingest(doc.id, pdf_file)

        await test_db.refresh(doc)
        assert doc.status == "failed"
        assert "API error" in doc.error_message

    async def test_pipeline_document_not_found(
        self,
        test_db: AsyncSession,
        mock_qdrant_for_pipeline: AsyncMock,
        mock_embedding: AsyncMock,
        pipeline_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline returns early if document ID doesn't exist."""
        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_for_pipeline,
            embedding_provider=mock_embedding,
            settings=pipeline_settings,
        )
        # Should not raise
        await pipeline.ingest("nonexistent-id", tmp_path / "test.pdf")

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_pipeline_creates_collection_if_missing(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_for_pipeline: AsyncMock,
        mock_embedding: AsyncMock,
        pipeline_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline creates Qdrant collection if it doesn't exist yet."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        mock_qdrant_for_pipeline.collection_exists.return_value = False

        doc = Document(
            collection_id="col-1",
            filename="test.pdf",
            status="pending",
        )
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_for_pipeline,
            embedding_provider=mock_embedding,
            settings=pipeline_settings,
        )
        await pipeline.ingest(doc.id, pdf_file)

        mock_qdrant_for_pipeline.create_collection.assert_called_once()

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_pipeline_with_overrides(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_for_pipeline: AsyncMock,
        mock_embedding: AsyncMock,
        pipeline_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline respects parser and chunking overrides."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "pymupdf4llm"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        doc = Document(
            collection_id="col-1",
            filename="test.pdf",
            status="pending",
        )
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_for_pipeline,
            embedding_provider=mock_embedding,
            settings=pipeline_settings,
        )
        await pipeline.ingest(
            doc.id,
            pdf_file,
            parser_override="pymupdf4llm",
            chunk_size_override=256,
            chunk_overlap_override=25,
        )

        mock_create_parser.assert_called_once_with("pymupdf4llm")
        await test_db.refresh(doc)
        assert doc.status == "complete"

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_pipeline_empty_document(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_for_pipeline: AsyncMock,
        mock_embedding: AsyncMock,
        pipeline_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline handles documents that produce no chunks."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = ParseResult(
            markdown="",
            metadata=DocumentMetadata(page_count=1),
            pages=[PageData(page_number=1)],
        )
        mock_create_parser.return_value = mock_parser

        doc = Document(
            collection_id="col-1",
            filename="empty.pdf",
            status="pending",
        )
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_for_pipeline,
            embedding_provider=mock_embedding,
            settings=pipeline_settings,
        )
        await pipeline.ingest(doc.id, pdf_file)

        await test_db.refresh(doc)
        assert doc.status == "complete"
        assert doc.chunk_count == 0
        mock_embedding.embed_texts.assert_not_called()
