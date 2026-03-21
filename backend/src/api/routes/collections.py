"""Collections API — CRUD for document collections."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.collection import Collection
from src.models.document import Document
from src.schemas import CollectionCreate, CollectionResponse, CollectionUpdate
from src.storage.database import get_db

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(body: CollectionCreate, db: AsyncSession = Depends(get_db)):
    """Create a new document collection."""
    collection = Collection(name=body.name, description=body.description)
    db.add(collection)
    await db.flush()
    await db.refresh(collection)
    return CollectionResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        document_count=0,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )


@router.get("", response_model=list[CollectionResponse])
async def list_collections(db: AsyncSession = Depends(get_db)):
    """List all collections with document counts."""
    stmt = (
        select(Collection, func.count(Document.id).label("doc_count"))
        .outerjoin(Document, Document.collection_id == Collection.id)
        .group_by(Collection.id)
        .order_by(Collection.created_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.all()
    return [
        CollectionResponse(
            id=col.id,
            name=col.name,
            description=col.description,
            document_count=doc_count,
            created_at=col.created_at,
            updated_at=col.updated_at,
        )
        for col, doc_count in rows
    ]


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(collection_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single collection by ID."""
    collection = await db.get(Collection, collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    doc_count_result = await db.execute(
        select(func.count(Document.id)).where(Document.collection_id == collection_id)
    )
    doc_count = doc_count_result.scalar() or 0
    return CollectionResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        document_count=doc_count,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )


@router.put("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: str, body: CollectionUpdate, db: AsyncSession = Depends(get_db)
):
    """Update a collection's name or description."""
    collection = await db.get(Collection, collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if body.name is not None:
        collection.name = body.name
    if body.description is not None:
        collection.description = body.description
    await db.flush()
    await db.refresh(collection)
    doc_count_result = await db.execute(
        select(func.count(Document.id)).where(Document.collection_id == collection_id)
    )
    doc_count = doc_count_result.scalar() or 0
    return CollectionResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        document_count=doc_count,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )


@router.delete("/{collection_id}", status_code=204)
async def delete_collection(collection_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a collection and all its documents."""
    collection = await db.get(Collection, collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    await db.delete(collection)
