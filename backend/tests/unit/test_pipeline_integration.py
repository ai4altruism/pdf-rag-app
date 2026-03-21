"""Integration tests for the full ingestion pipeline with enrichment."""

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
def enrichment_settings() -> Settings:
    return Settings(
        parser="docling",
        chunk_size_tokens=512,
        chunk_overlap_tokens=50,
        contextual_enrichment=True,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=1024,
    )


@pytest.fixture
def no_enrichment_settings() -> Settings:
    return Settings(
        parser="docling",
        chunk_size_tokens=512,
        chunk_overlap_tokens=50,
        contextual_enrichment=False,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=1024,
    )


@pytest.fixture
def mock_qdrant_store() -> AsyncMock:
    mock = AsyncMock()
    mock.collection_exists.return_value = True
    mock.create_collection.return_value = None
    mock.delete_points_by_document.return_value = None
    mock.upsert_points.return_value = None
    return mock


@pytest.fixture
def mock_embedding() -> AsyncMock:
    mock = AsyncMock()
    mock.embed_texts.return_value = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    mock.model_name = "text-embedding-3-large"
    mock.dimensions = 1024
    return mock


@pytest.fixture
def mock_llm() -> AsyncMock:
    mock = AsyncMock()
    mock.generate.return_value = "This chunk provides context about the topic."
    mock.model_name = "test-llm"
    return mock


def _make_parse_result(with_tables: bool = False) -> ParseResult:
    if with_tables:
        markdown = (
            "# Report\n\nIntroduction text.\n\n"
            "## Data\n\n| Col A | Col B |\n|---|---|\n| 1 | 2 |\n\n"
            "## Conclusion\n\nFinal thoughts."
        )
    else:
        markdown = (
            "# Introduction\n\nFirst paragraph of the document.\n\n"
            "## Methods\n\nDescription of methods used.\n\n"
            "## Results\n\nKey findings here."
        )
    return ParseResult(
        markdown=markdown,
        metadata=DocumentMetadata(
            title="Test Document", author="Test Author", language="en", page_count=3
        ),
        pages=[PageData(page_number=i) for i in range(1, 4)],
    )


class TestPipelineWithEnrichment:
    """Integration tests for the pipeline with contextual enrichment enabled."""

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_enrichment_prepends_summaries(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_store: AsyncMock,
        mock_embedding: AsyncMock,
        mock_llm: AsyncMock,
        enrichment_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline enriches chunks before embedding when enabled."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        doc = Document(collection_id="col-1", filename="test.pdf", status="pending")
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_store,
            embedding_provider=mock_embedding,
            settings=enrichment_settings,
            llm_provider=mock_llm,
        )
        await pipeline.ingest(doc.id, pdf_file)

        await test_db.refresh(doc)
        assert doc.status == "complete"

        # Verify LLM was called for enrichment (once per child chunk)
        assert mock_llm.generate.call_count > 0

        # Verify the texts passed to embed_texts contain enrichment prefix
        embed_call = mock_embedding.embed_texts.call_args
        texts = embed_call.args[0]
        for text in texts:
            assert "This chunk provides context" in text

        # Verify ingestion job records enrichment was enabled
        result = await test_db.execute(
            select(IngestionJob).where(IngestionJob.document_id == doc.id)
        )
        job = result.scalar_one()
        assert job.contextual_enrichment is True

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_enrichment_skipped_when_disabled(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_store: AsyncMock,
        mock_embedding: AsyncMock,
        mock_llm: AsyncMock,
        no_enrichment_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline skips enrichment when disabled in settings."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        doc = Document(collection_id="col-1", filename="test.pdf", status="pending")
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_store,
            embedding_provider=mock_embedding,
            settings=no_enrichment_settings,
            llm_provider=mock_llm,
        )
        await pipeline.ingest(doc.id, pdf_file)

        await test_db.refresh(doc)
        assert doc.status == "complete"
        mock_llm.generate.assert_not_called()

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_enrichment_skipped_when_no_llm(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_store: AsyncMock,
        mock_embedding: AsyncMock,
        enrichment_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline skips enrichment when no LLM provider is available."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        doc = Document(collection_id="col-1", filename="test.pdf", status="pending")
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        # No LLM provider
        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_store,
            embedding_provider=mock_embedding,
            settings=enrichment_settings,
            llm_provider=None,
        )
        await pipeline.ingest(doc.id, pdf_file)

        await test_db.refresh(doc)
        assert doc.status == "complete"

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_pipeline_with_table_content(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_store: AsyncMock,
        mock_embedding: AsyncMock,
        mock_llm: AsyncMock,
        enrichment_settings: Settings,
        tmp_path: Path,
    ):
        """Pipeline handles documents with tables correctly."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result(with_tables=True)
        mock_create_parser.return_value = mock_parser

        doc = Document(collection_id="col-1", filename="report.pdf", status="pending")
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_store,
            embedding_provider=mock_embedding,
            settings=enrichment_settings,
            llm_provider=mock_llm,
        )
        await pipeline.ingest(doc.id, pdf_file)

        await test_db.refresh(doc)
        assert doc.status == "complete"
        assert doc.chunk_count > 0

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_qdrant_payloads_have_correct_fields(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_store: AsyncMock,
        mock_embedding: AsyncMock,
        no_enrichment_settings: Settings,
        tmp_path: Path,
    ):
        """Verify Qdrant point payloads contain all required fields."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        doc = Document(collection_id="col-1", filename="test.pdf", status="pending")
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_store,
            embedding_provider=mock_embedding,
            settings=no_enrichment_settings,
        )
        await pipeline.ingest(doc.id, pdf_file)

        # Inspect the payloads passed to upsert_points
        call_kwargs = mock_qdrant_store.upsert_points.call_args.kwargs
        payloads = call_kwargs["payloads"]

        required_fields = {
            "document_id", "filename", "page_numbers", "header_chain",
            "chunk_index", "chunk_token_count", "chunk_text",
            "parent_chunk_id", "embedding_model",
        }
        for payload in payloads:
            assert required_fields.issubset(payload.keys())
            assert payload["document_id"] == doc.id
            assert payload["filename"] == "test.pdf"
            assert payload["embedding_model"] == "text-embedding-3-large"
            assert isinstance(payload["page_numbers"], list)
            assert isinstance(payload["header_chain"], list)

    @patch("src.pipelines.ingestion.pipeline.create_parser")
    async def test_parent_child_links_in_payloads(
        self,
        mock_create_parser,
        test_db: AsyncSession,
        mock_qdrant_store: AsyncMock,
        mock_embedding: AsyncMock,
        no_enrichment_settings: Settings,
        tmp_path: Path,
    ):
        """Verify child chunks have parent_chunk_id set."""
        mock_parser = MagicMock()
        mock_parser.parser_name = "docling"
        mock_parser.parse.return_value = _make_parse_result()
        mock_create_parser.return_value = mock_parser

        doc = Document(collection_id="col-1", filename="test.pdf", status="pending")
        test_db.add(doc)
        await test_db.commit()
        await test_db.refresh(doc)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        pipeline = IngestionPipeline(
            db=test_db,
            qdrant=mock_qdrant_store,
            embedding_provider=mock_embedding,
            settings=no_enrichment_settings,
        )
        await pipeline.ingest(doc.id, pdf_file)

        call_kwargs = mock_qdrant_store.upsert_points.call_args.kwargs
        payloads = call_kwargs["payloads"]

        # All stored chunks are children — they should have a parent_chunk_id
        for payload in payloads:
            assert payload["parent_chunk_id"] != ""
