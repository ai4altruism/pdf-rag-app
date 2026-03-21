"""Ingestion pipeline — orchestrates parse → chunk → embed → store."""

import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.collections import _qdrant_collection_name
from src.config import Settings
from src.logging import get_logger
from src.models.document import Document
from src.models.ingestion_job import IngestionJob
from src.pipelines.ingestion import create_parser
from src.pipelines.ingestion.chunker import DocumentChunker
from src.providers.base import EmbeddingProvider
from src.storage.qdrant import ChunkPayload, QdrantStore

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates end-to-end document ingestion.

    Pipeline stages:
    1. Parse — convert PDF to Markdown via configured parser
    2. Chunk — split into parent/child chunks with metadata
    3. Embed — generate dense vectors for child chunks
    4. Store — upsert into Qdrant with full payload

    Each stage updates the ingestion job and document status. On failure,
    rolls back Qdrant points for atomicity.
    """

    def __init__(
        self,
        db: AsyncSession,
        qdrant: QdrantStore,
        embedding_provider: EmbeddingProvider,
        settings: Settings,
    ):
        self._db = db
        self._qdrant = qdrant
        self._embedding = embedding_provider
        self._settings = settings

    async def ingest(
        self,
        document_id: str,
        file_path: Path,
        *,
        parser_override: str | None = None,
        chunk_size_override: int | None = None,
        chunk_overlap_override: int | None = None,
    ) -> None:
        """Run the full ingestion pipeline for a single document.

        Args:
            document_id: ID of the Document record in the database.
            file_path: Path to the uploaded PDF on disk.
            parser_override: Override the configured parser type.
            chunk_size_override: Override the configured chunk size.
            chunk_overlap_override: Override the configured chunk overlap.
        """
        doc = await self._db.get(Document, document_id)
        if not doc:
            logger.error("document_not_found", document_id=document_id)
            return

        parser_type = parser_override or self._settings.parser
        chunk_size = chunk_size_override or self._settings.chunk_size_tokens
        chunk_overlap = chunk_overlap_override or self._settings.chunk_overlap_tokens

        # Create ingestion job
        job = IngestionJob(
            document_id=document_id,
            status="running",
            parser=parser_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            contextual_enrichment=False,  # Sprint 4
            started_at=datetime.utcnow(),
        )
        self._db.add(job)
        doc.status = "parsing"
        await self._db.commit()

        qdrant_name = _qdrant_collection_name(doc.collection_id)

        try:
            # --- Stage 1: Parse ---
            logger.info("ingestion_stage", stage="parsing", document_id=document_id)
            parser = create_parser(parser_type)
            parse_result = parser.parse(file_path)

            doc.title = parse_result.metadata.title
            doc.author = parse_result.metadata.author
            doc.language = parse_result.metadata.language
            doc.parser_used = parser.parser_name
            if parse_result.metadata.page_count:
                doc.page_count = parse_result.metadata.page_count
            doc.status = "chunking"
            await self._db.commit()

            # --- Stage 2: Chunk ---
            logger.info("ingestion_stage", stage="chunking", document_id=document_id)
            chunker = DocumentChunker(
                chunk_size_tokens=chunk_size,
                chunk_overlap_tokens=chunk_overlap,
            )
            page_numbers = [p.page_number for p in parse_result.pages] or list(
                range(1, (doc.page_count or 0) + 1)
            )
            chunks = chunker.chunk(parse_result.markdown, page_numbers=page_numbers)
            child_chunks = [c for c in chunks if not c.is_parent]

            if not child_chunks:
                doc.status = "complete"
                doc.chunk_count = 0
                job.status = "complete"
                job.completed_at = datetime.utcnow()
                await self._db.commit()
                return

            doc.status = "embedding"
            await self._db.commit()

            # --- Stage 3: Embed ---
            logger.info(
                "ingestion_stage",
                stage="embedding",
                document_id=document_id,
                chunk_count=len(child_chunks),
            )
            texts = [c.text for c in child_chunks]
            embeddings = await self._embedding.embed_texts(texts)

            doc.status = "storing"
            await self._db.commit()

            # --- Stage 4: Store ---
            logger.info("ingestion_stage", stage="storing", document_id=document_id)

            # Delete old points first (for re-ingestion)
            if await self._qdrant.collection_exists(qdrant_name):
                await self._qdrant.delete_points_by_document(qdrant_name, document_id)
            else:
                await self._qdrant.create_collection(qdrant_name)

            point_ids = [str(uuid.uuid4()) for _ in child_chunks]
            payloads = []
            for chunk in child_chunks:
                payload = ChunkPayload(
                    document_id=document_id,
                    filename=doc.filename,
                    page_numbers=chunk.page_numbers,
                    header_chain=chunk.header_chain,
                    chunk_index=chunk.chunk_index,
                    chunk_token_count=chunk.token_count,
                    chunk_text=chunk.text,
                    parent_chunk_id=chunk.parent_chunk_id,
                    embedding_model=self._embedding.model_name,
                )
                payloads.append(payload.__dict__)

            await self._qdrant.upsert_points(
                collection_name=qdrant_name,
                ids=point_ids,
                dense_vectors=embeddings,
                texts=texts,
                payloads=payloads,
            )

            # --- Done ---
            doc.status = "complete"
            doc.chunk_count = len(child_chunks)
            doc.embedding_model = self._embedding.model_name
            doc.error_message = None
            job.status = "complete"
            job.completed_at = datetime.utcnow()
            await self._db.commit()

            logger.info(
                "ingestion_complete",
                document_id=document_id,
                chunk_count=len(child_chunks),
                parser=parser.parser_name,
            )

        except Exception as e:
            logger.error(
                "ingestion_failed",
                document_id=document_id,
                error=str(e),
            )
            # Rollback Qdrant points on failure
            try:
                if await self._qdrant.collection_exists(qdrant_name):
                    await self._qdrant.delete_points_by_document(qdrant_name, document_id)
            except Exception:
                logger.error("rollback_failed", document_id=document_id)

            doc.status = "failed"
            doc.error_message = str(e)
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await self._db.commit()
