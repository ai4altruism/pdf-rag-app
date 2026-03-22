"""Tests for API endpoints."""


class TestHealthEndpoint:
    async def test_health_returns_status(self, client):
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "qdrant" in data
        assert "embedding_provider" in data
        assert "llm_provider" in data

    async def test_health_qdrant_ok(self, client, mock_qdrant):
        mock_qdrant.health_check.return_value = True
        response = await client.get("/api/health")
        data = response.json()
        assert data["qdrant"]["status"] == "ok"

    async def test_health_qdrant_error(self, client, mock_qdrant):
        mock_qdrant.health_check.return_value = False
        response = await client.get("/api/health")
        data = response.json()
        assert data["qdrant"]["status"] == "error"


class TestCollectionsAPI:
    async def test_create_collection(self, client):
        response = await client.post(
            "/api/collections",
            json={"name": "Test Collection", "description": "Testing"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Collection"
        assert data["description"] == "Testing"
        assert data["document_count"] == 0
        assert "id" in data

    async def test_list_collections(self, client):
        await client.post("/api/collections", json={"name": "Col 1"})
        await client.post("/api/collections", json={"name": "Col 2"})
        response = await client.get("/api/collections")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    async def test_get_collection(self, client):
        create_resp = await client.post("/api/collections", json={"name": "Get Me"})
        col_id = create_resp.json()["id"]
        response = await client.get(f"/api/collections/{col_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "Get Me"

    async def test_get_collection_not_found(self, client):
        response = await client.get("/api/collections/nonexistent")
        assert response.status_code == 404

    async def test_update_collection(self, client):
        create_resp = await client.post("/api/collections", json={"name": "Old Name"})
        col_id = create_resp.json()["id"]
        response = await client.put(
            f"/api/collections/{col_id}",
            json={"name": "New Name"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "New Name"

    async def test_delete_collection(self, client):
        create_resp = await client.post("/api/collections", json={"name": "Delete Me"})
        col_id = create_resp.json()["id"]
        response = await client.delete(f"/api/collections/{col_id}")
        assert response.status_code == 204
        get_resp = await client.get(f"/api/collections/{col_id}")
        assert get_resp.status_code == 404


class TestSettingsAPI:
    async def test_get_settings(self, client):
        response = await client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert data["embedding_provider"] == "openai"
        assert data["embedding_model"] == "text-embedding-3-large"
        assert data["llm_provider"] == "anthropic"


class TestQueryAPI:
    def _patch_providers(self, mock_embedding):
        """Patch all provider factories used by the query route."""
        from contextlib import ExitStack
        from unittest.mock import patch

        stack = ExitStack()
        stack.enter_context(
            patch("src.api.routes.query.create_embedding_provider", return_value=mock_embedding)
        )
        stack.enter_context(
            patch("src.api.routes.query.create_llm_provider", side_effect=ValueError("no key"))
        )
        stack.enter_context(
            patch(
                "src.api.routes.query.create_reranker_provider",
                side_effect=ValueError("no key"),
            )
        )
        return stack

    async def test_query_returns_results(self, client, mock_qdrant):
        from unittest.mock import AsyncMock

        from src.storage.qdrant import SearchResult

        # Create a collection
        col_resp = await client.post("/api/collections", json={"name": "Query Test"})
        col_id = col_resp.json()["id"]

        # Mock embedding provider
        mock_provider = AsyncMock()
        mock_provider.embed_query.return_value = [0.1] * 1024

        # Mock hybrid search to return a result
        mock_qdrant.hybrid_search.return_value = [
            SearchResult(
                id="point-1",
                score=0.95,
                payload={
                    "document_id": "doc-1",
                    "filename": "test.pdf",
                    "page_numbers": [1],
                    "header_chain": ["Section 1"],
                    "chunk_text": "Test chunk content",
                    "parent_chunk_id": "",
                    "chunk_index": 0,
                },
            )
        ]

        with self._patch_providers(mock_provider):
            response = await client.post(
                "/api/query",
                json={
                    "query": "test question",
                    "collection_ids": [col_id],
                    "options": {"multi_query": False, "context_expansion": "off"},
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "test.pdf"
        assert data["sources"][0]["relevance_score"] == 0.95
        assert data["metadata"]["retrieval_count"] == 1

    async def test_query_empty_collection(self, client, mock_qdrant):
        from unittest.mock import AsyncMock

        col_resp = await client.post("/api/collections", json={"name": "Empty"})
        col_id = col_resp.json()["id"]

        mock_provider = AsyncMock()
        mock_provider.embed_query.return_value = [0.1] * 1024
        mock_qdrant.hybrid_search.return_value = []

        with self._patch_providers(mock_provider):
            response = await client.post(
                "/api/query",
                json={
                    "query": "test question",
                    "collection_ids": [col_id],
                    "options": {"multi_query": False, "context_expansion": "off"},
                },
            )
        assert response.status_code == 200
        assert response.json()["sources"] == []

    async def test_query_response_metadata_shape(self, client, mock_qdrant):
        from unittest.mock import AsyncMock

        col_resp = await client.post("/api/collections", json={"name": "Meta Test"})
        col_id = col_resp.json()["id"]

        mock_provider = AsyncMock()
        mock_provider.embed_query.return_value = [0.1] * 1024
        mock_qdrant.hybrid_search.return_value = []

        with self._patch_providers(mock_provider):
            response = await client.post(
                "/api/query",
                json={
                    "query": "test",
                    "collection_ids": [col_id],
                    "options": {"multi_query": False, "context_expansion": "off"},
                },
            )
        meta = response.json()["metadata"]
        assert "retrieval_count" in meta
        assert "reranked_count" in meta
        assert "corrective_rag_triggered" in meta
        assert "query_variations" in meta
        assert "latency_ms" in meta
        assert "model_used" in meta


class TestConversationsAPI:
    async def test_create_conversation(self, client):
        col_resp = await client.post("/api/collections", json={"name": "Conv Test"})
        col_id = col_resp.json()["id"]

        response = await client.post(
            "/api/conversations",
            json={"collection_id": col_id, "title": "Test Chat"},
        )
        assert response.status_code == 201
        assert response.json()["title"] == "Test Chat"

    async def test_list_conversations(self, client):
        col_resp = await client.post("/api/collections", json={"name": "Conv List"})
        col_id = col_resp.json()["id"]
        await client.post("/api/conversations", json={"collection_id": col_id})
        response = await client.get("/api/conversations")
        assert response.status_code == 200
        assert len(response.json()) >= 1

    async def test_delete_conversation(self, client):
        col_resp = await client.post("/api/collections", json={"name": "Conv Del"})
        col_id = col_resp.json()["id"]
        conv_resp = await client.post("/api/conversations", json={"collection_id": col_id})
        conv_id = conv_resp.json()["id"]
        response = await client.delete(f"/api/conversations/{conv_id}")
        assert response.status_code == 204
