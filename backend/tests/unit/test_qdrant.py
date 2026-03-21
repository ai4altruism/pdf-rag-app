"""Tests for QdrantStore wrapper."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from src.storage.qdrant import ChunkPayload, QdrantStore, SearchResult


class TestSearchResult:
    def test_search_result_dataclass(self):
        r = SearchResult(id="abc", score=0.95, payload={"key": "val"})
        assert r.id == "abc"
        assert r.score == 0.95
        assert r.payload == {"key": "val"}


class TestChunkPayload:
    def test_chunk_payload_defaults(self):
        p = ChunkPayload(document_id="d1", filename="test.pdf")
        assert p.document_id == "d1"
        assert p.page_numbers == []
        assert p.header_chain == []
        assert p.chunk_index == 0
        assert p.chunk_text == ""
        assert p.embedding_model == ""


class TestQdrantStore:
    @pytest.fixture
    def mock_client(self):
        return AsyncMock()

    @pytest.fixture
    def store(self, mock_client):
        s = QdrantStore(url="http://localhost:6333", dense_dim=1024)
        s._client = mock_client
        return s

    async def test_create_collection_new(self, store, mock_client):
        mock_client.collection_exists.return_value = False
        await store.create_collection("test_col")
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "test_col"

    async def test_create_collection_exists(self, store, mock_client):
        mock_client.collection_exists.return_value = True
        await store.create_collection("test_col")
        mock_client.create_collection.assert_not_called()

    async def test_delete_collection_exists(self, store, mock_client):
        mock_client.collection_exists.return_value = True
        await store.delete_collection("test_col")
        mock_client.delete_collection.assert_called_once_with("test_col")

    async def test_delete_collection_not_exists(self, store, mock_client):
        mock_client.collection_exists.return_value = False
        await store.delete_collection("test_col")
        mock_client.delete_collection.assert_not_called()

    async def test_collection_exists(self, store, mock_client):
        mock_client.collection_exists.return_value = True
        assert await store.collection_exists("col") is True
        mock_client.collection_exists.return_value = False
        assert await store.collection_exists("col") is False

    async def test_upsert_points(self, store, mock_client):
        await store.upsert_points(
            collection_name="test_col",
            ids=["id1", "id2"],
            dense_vectors=[[0.1] * 1024, [0.2] * 1024],
            texts=["text one", "text two"],
            payloads=[{"key": "a"}, {"key": "b"}],
        )
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_col"
        assert len(call_kwargs["points"]) == 2

    async def test_upsert_points_batched(self, store, mock_client):
        """Upsert with >100 points should batch into multiple calls."""
        n = 150
        await store.upsert_points(
            collection_name="test_col",
            ids=[f"id{i}" for i in range(n)],
            dense_vectors=[[0.1] * 1024 for _ in range(n)],
            texts=[f"text {i}" for i in range(n)],
            payloads=[{"idx": i} for i in range(n)],
        )
        assert mock_client.upsert.call_count == 2

    async def test_delete_points_by_document(self, store, mock_client):
        await store.delete_points_by_document("test_col", "doc-123")
        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_col"

    async def test_hybrid_search(self, store, mock_client):
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.9
        mock_point.payload = {"document_id": "d1", "chunk_text": "hello"}

        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_client.query_points.return_value = mock_result

        results = await store.hybrid_search(
            collection_name="test_col",
            query_dense=[0.1] * 1024,
            query_text="test query",
            top_k=10,
        )
        assert len(results) == 1
        assert results[0].id == "p1"
        assert results[0].score == 0.9
        assert results[0].payload["chunk_text"] == "hello"
        mock_client.query_points.assert_called_once()

    async def test_hybrid_search_with_filters(self, store, mock_client):
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result

        await store.hybrid_search(
            collection_name="test_col",
            query_dense=[0.1] * 1024,
            query_text="test",
            filters={"embedding_model": "text-embedding-3-large"},
        )
        call_kwargs = mock_client.query_points.call_args.kwargs
        # Prefetch items should have filter set
        prefetch = call_kwargs["prefetch"]
        assert len(prefetch) == 2
        assert prefetch[0].filter is not None

    async def test_dense_search(self, store, mock_client):
        mock_point = MagicMock()
        mock_point.id = "p2"
        mock_point.score = 0.85
        mock_point.payload = {"chunk_text": "dense result"}

        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_client.query_points.return_value = mock_result

        results = await store.dense_search(
            collection_name="test_col",
            query_dense=[0.1] * 1024,
            top_k=5,
        )
        assert len(results) == 1
        assert results[0].score == 0.85

    async def test_health_check_ok(self, store, mock_client):
        mock_client.get_collections.return_value = MagicMock()
        assert await store.health_check() is True

    async def test_health_check_fail(self, store, mock_client):
        mock_client.get_collections.side_effect = Exception("connection refused")
        assert await store.health_check() is False

    async def test_get_points_count(self, store, mock_client):
        mock_info = MagicMock()
        mock_info.points_count = 42
        mock_client.get_collection.return_value = mock_info
        count = await store.get_points_count("test_col")
        assert count == 42

    def test_build_filter(self, store):
        f = store._build_filter({"embedding_model": "test", "document_id": "d1"})
        assert isinstance(f, models.Filter)
        assert len(f.must) == 2

    def test_build_filter_skips_none(self, store):
        f = store._build_filter({"embedding_model": "test", "document_id": None})
        assert len(f.must) == 1

    def test_build_filter_all_none(self, store):
        f = store._build_filter({"a": None, "b": None})
        assert f is None

    def test_get_sparse_encoder(self, store):
        encoder = store.get_sparse_encoder()
        from src.storage.sparse import BM25SparseEncoder
        assert isinstance(encoder, BM25SparseEncoder)
