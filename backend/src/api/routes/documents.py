"""Documents API — upload, list, get, delete PDFs."""

import hashlib
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.routes.collections import _qdrant_collection_name
from src.config import Settings, get_settings
from src.models.document import Document
from src.providers.base import EmbeddingProvider
from src.schemas import DocumentResponse, ReingestRequest
from src.storage import get_embedding_provider, get_qdrant
from src.storage.database import get_db
from src.storage.qdrant import QdrantStore

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}


async def _run_ingestion(
    document_id: str,
    file_path: Path,
    settings: Settings,
    qdrant: QdrantStore,
    embedding_provider: EmbeddingProvider,
    parser_override: str | None = None,
    chunk_size_override: int | None = None,
    chunk_overlap_override: int | None = None,
) -> None:
    """Run ingestion in a background task with its own DB session."""
    from src.pipelines.ingestion.pipeline import IngestionPipeline

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as db:
        try:
            pipeline = IngestionPipeline(
                db=db, qdrant=qdrant, embedding_provider=embedding_provider, settings=settings
            )
            await pipeline.ingest(
                document_id,
                file_path,
                parser_override=parser_override,
                chunk_size_override=chunk_size_override,
                chunk_overlap_override=chunk_overlap_override,
            )
        except Exception:
            await db.rollback()
            raise

    await engine.dispose()


@router.post("/upload", response_model=list[DocumentResponse], status_code=201)
async def upload_documents(
    files: list[UploadFile],
    collection_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
):
    """Upload one or more PDF files to a collection."""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Only PDF files are accepted.",
            )

        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds {settings.max_upload_size_mb}MB limit.",
            )

        # Sanitize filename
        safe_name = Path(file.filename).name if file.filename else "upload.pdf"
        file_hash = hashlib.sha256(content).hexdigest()

        # Save to disk
        file_path = upload_dir / f"{file_hash}_{safe_name}"
        file_path.write_bytes(content)

        # Extract page count
        page_count = _get_pdf_page_count(file_path)

        doc = Document(
            collection_id=collection_id,
            filename=safe_name,
            file_size_bytes=len(content),
            file_hash=file_hash,
            page_count=page_count,
            status="pending",
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        documents.append((doc, file_path))

    # Schedule background ingestion for each document
    response = []
    for doc, fpath in documents:
        background_tasks.add_task(
            _run_ingestion,
            document_id=doc.id,
            file_path=fpath,
            settings=settings,
            qdrant=qdrant,
            embedding_provider=embedding_provider,
        )
        response.append(DocumentResponse.model_validate(doc))

    return response


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    collection_id: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List documents, optionally filtered by collection and/or status."""
    stmt = select(Document).order_by(Document.created_at.desc())
    if collection_id:
        stmt = stmt.where(Document.collection_id == collection_id)
    if status:
        stmt = stmt.where(Document.status == status)
    result = await db.execute(stmt)
    return [DocumentResponse.model_validate(doc) for doc in result.scalars().all()]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single document by ID."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantStore = Depends(get_qdrant),
):
    """Delete a document and its vectors from Qdrant."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete vectors from Qdrant
    qdrant_name = _qdrant_collection_name(doc.collection_id)
    if await qdrant.collection_exists(qdrant_name):
        await qdrant.delete_points_by_document(qdrant_name, document_id)

    await db.delete(doc)


@router.post("/{document_id}/reingest", response_model=DocumentResponse)
async def reingest_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    body: ReingestRequest | None = None,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
):
    """Re-process a document with optional new settings."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.status = "pending"
    await db.flush()
    await db.refresh(doc)

    # Resolve file path
    upload_dir = Path(settings.upload_dir)
    file_path = _find_document_file(upload_dir, doc.file_hash, doc.filename)
    if not file_path:
        raise HTTPException(status_code=404, detail="Source PDF file not found on disk")

    # Parse overrides from request body
    parser_override = body.parser if body else None
    chunk_size_override = body.chunk_size_tokens if body else None
    chunk_overlap_override = body.chunk_overlap_tokens if body else None

    background_tasks.add_task(
        _run_ingestion,
        document_id=doc.id,
        file_path=file_path,
        settings=settings,
        qdrant=qdrant,
        embedding_provider=embedding_provider,
        parser_override=parser_override,
        chunk_size_override=chunk_size_override,
        chunk_overlap_override=chunk_overlap_override,
    )

    return DocumentResponse.model_validate(doc)


def _find_document_file(upload_dir: Path, file_hash: str | None, filename: str) -> Path | None:
    """Locate the uploaded PDF file on disk by hash and filename."""
    if not file_hash:
        return None
    expected = upload_dir / f"{file_hash}_{filename}"
    if expected.exists():
        return expected
    return None


def _get_pdf_page_count(file_path: Path) -> int | None:
    """Extract page count from a PDF file."""
    try:
        import pypdf

        reader = pypdf.PdfReader(str(file_path))
        return len(reader.pages)
    except Exception:
        return None
