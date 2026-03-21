"""Tests for API endpoints."""



class TestHealthEndpoint:
    async def test_health_returns_ok(self, client):
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "qdrant" in data
        assert "embedding_provider" in data
        assert "llm_provider" in data


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
    async def test_query_stub(self, client):
        # First create a collection to use its ID
        col_resp = await client.post("/api/collections", json={"name": "Query Test"})
        col_id = col_resp.json()["id"]

        response = await client.post(
            "/api/query",
            json={"query": "test question", "collection_ids": [col_id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "metadata" in data


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
