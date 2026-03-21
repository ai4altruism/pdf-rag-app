"""Qdrant vector store wrapper with hybrid search support."""

from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient, models

from src.logging import get_logger
from src.storage.sparse import BM25SparseEncoder

logger = get_logger(__name__)


@dataclass
class ChunkPayload:
    """Payload schema for a stored chunk in Qdrant."""

    document_id: str
    filename: str
    page_numbers: list[int] = field(default_factory=list)
    header_chain: list[str] = field(default_factory=list)
    chunk_index: int = 0
    chunk_token_count: int = 0
    chunk_text: str = ""
    contextual_summary: str = ""
    parent_chunk_id: str = ""
    embedding_model: str = ""


@dataclass
class SearchResult:
    """A single search result from Qdrant."""

    id: str
    score: float
    payload: dict


class QdrantStore:
    """Wrapper around Qdrant for vector storage and hybrid search.

    Each collection has two named vectors:
    - "dense": HNSW-indexed dense embeddings
    - "sparse": Sparse BM25-style vectors
    """

    def __init__(self, url: str = "http://localhost:6333", dense_dim: int = 1024):
        self._client = AsyncQdrantClient(url=url)
        self._dense_dim = dense_dim
        self._sparse_encoder = BM25SparseEncoder()

    async def create_collection(self, name: str) -> None:
        """Create a Qdrant collection with dense + sparse vector config."""
        exists = await self._client.collection_exists(name)
        if exists:
            logger.info("collection_exists", name=name)
            return

        await self._client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        logger.info("collection_created", name=name, dense_dim=self._dense_dim)

    async def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection entirely."""
        exists = await self._client.collection_exists(name)
        if exists:
            await self._client.delete_collection(name)
            logger.info("collection_deleted", name=name)

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        return await self._client.collection_exists(name)

    async def upsert_points(
        self,
        collection_name: str,
        ids: list[str],
        dense_vectors: list[list[float]],
        texts: list[str],
        payloads: list[dict],
    ) -> None:
        """Insert or update points with both dense and sparse vectors."""
        sparse_vectors = self._sparse_encoder.encode_batch(texts)

        points = []
        for point_id, dense_vec, sparse_vec, payload in zip(
            ids, dense_vectors, sparse_vectors, payloads, strict=True
        ):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                    },
                    payload=payload,
                )
            )

        # Batch upsert in groups of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.info(
            "points_upserted",
            collection=collection_name,
            count=len(points),
        )

    async def delete_points_by_document(
        self, collection_name: str, document_id: str
    ) -> None:
        """Delete all points belonging to a specific document."""
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.info(
            "points_deleted",
            collection=collection_name,
            document_id=document_id,
        )

    async def hybrid_search(
        self,
        collection_name: str,
        query_dense: list[float],
        query_text: str,
        top_k: int = 20,
        rrf_k: int = 60,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search combining dense and sparse retrieval with RRF fusion.

        Args:
            collection_name: Qdrant collection to search
            query_dense: Dense embedding vector for the query
            query_text: Raw query text for sparse BM25 encoding
            top_k: Number of results to return
            rrf_k: RRF fusion parameter (higher = more weight to lower-ranked results)
            filters: Optional metadata filters {"document_id": "...", "embedding_model": "..."}
        """
        query_sparse = self._sparse_encoder.encode(query_text)

        # Build Qdrant filter if provided
        qdrant_filter = self._build_filter(filters) if filters else None

        prefetch_limit = top_k * 2  # over-fetch for better fusion

        results = await self._client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices,
                        values=query_sparse.values,
                    ),
                    using="sparse",
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        search_results = []
        for point in results.points:
            search_results.append(
                SearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload or {},
                )
            )

        logger.info(
            "hybrid_search_complete",
            collection=collection_name,
            result_count=len(search_results),
            top_score=search_results[0].score if search_results else 0.0,
        )
        return search_results

    async def dense_search(
        self,
        collection_name: str,
        query_dense: list[float],
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Dense-only vector similarity search (fallback when hybrid is disabled)."""
        qdrant_filter = self._build_filter(filters) if filters else None

        results = await self._client.query_points(
            collection_name=collection_name,
            query=query_dense,
            using="dense",
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            SearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in results.points
        ]

    async def get_points_count(self, collection_name: str) -> int:
        """Get the number of points in a collection."""
        info = await self._client.get_collection(collection_name)
        return info.points_count

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    def _build_filter(self, filters: dict) -> models.Filter:
        """Build a Qdrant filter from a dict of field conditions."""
        conditions = []
        for key, value in filters.items():
            if value is not None:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=conditions) if conditions else None

    def get_sparse_encoder(self) -> BM25SparseEncoder:
        """Expose the sparse encoder for external use."""
        return self._sparse_encoder
