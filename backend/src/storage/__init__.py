"""Storage layer — Qdrant and database dependencies."""

from functools import lru_cache

from src.config import get_settings
from src.providers.base import EmbeddingProvider
from src.storage.qdrant import QdrantStore


@lru_cache
def get_qdrant() -> QdrantStore:
    """Get a cached QdrantStore instance."""
    settings = get_settings()
    return QdrantStore(
        url=settings.qdrant_url,
        dense_dim=settings.embedding_dimensions,
    )


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Get a cached EmbeddingProvider instance."""
    from src.providers import create_embedding_provider

    return create_embedding_provider(get_settings())
