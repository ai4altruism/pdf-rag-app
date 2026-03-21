"""Documents API — upload, list, get, delete PDFs."""

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings, get_settings
from src.models.document import Document
from src.schemas import DocumentResponse, ReingestRequest
from src.storage.database import get_db

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}


@router.post("/upload", response_model=list[DocumentResponse], status_code=201)
async def upload_documents(
    files: list[UploadFile],
    collection_id: str,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
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

        doc = Document(
            collection_id=collection_id,
            filename=safe_name,
            file_size_bytes=len(content),
            file_hash=file_hash,
            status="pending",
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        documents.append(doc)

    return [DocumentResponse.model_validate(doc) for doc in documents]


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
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a document and its associated data."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    # TODO: Also delete vectors from Qdrant (Sprint 2)
    await db.delete(doc)


@router.post("/{document_id}/reingest", response_model=DocumentResponse)
async def reingest_document(
    document_id: str,
    body: ReingestRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Re-process a document with optional new settings."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.status = "pending"
    await db.flush()
    await db.refresh(doc)
    # TODO: Trigger background ingestion pipeline (Sprint 3)
    return DocumentResponse.model_validate(doc)
