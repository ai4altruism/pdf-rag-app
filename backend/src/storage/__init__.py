"""Storage layer — Qdrant and database dependencies."""

from functools import lru_cache

from src.config import get_settings
from src.storage.qdrant import QdrantStore


@lru_cache
def get_qdrant() -> QdrantStore:
    """Get a cached QdrantStore instance."""
    settings = get_settings()
    return QdrantStore(
        url=settings.qdrant_url,
        dense_dim=settings.embedding_dimensions,
    )
