"""Tests for WebSocket streaming endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from src.config import Settings
from src.main import app
from src.storage.qdrant import SearchResult


@pytest.fixture
def test_settings():
    return Settings(
        embedding_provider="openai",
        embedding_model="test-model",
        embedding_dimensions=1024,
        llm_provider="anthropic",
        llm_model="test-model",
        reranker_provider="cohere",
        reranker_model="test-model",
    )


class TestWebSocketQuery:
    def test_ws_missing_fields(self):
        """WebSocket returns error for missing required fields."""
        client = TestClient(app)
        with client.websocket_connect("/ws/query") as ws:
            ws.send_json({"query": "test"})  # missing collection_ids
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "Missing required fields" in response["content"]

    def test_ws_empty_query(self):
        """WebSocket returns error for empty query."""
        client = TestClient(app)
        with client.websocket_connect("/ws/query") as ws:
            ws.send_json({"query": "", "collection_ids": ["col1"]})
            response = ws.receive_json()
            assert response["type"] == "error"

    @patch("src.api.websocket.create_reranker_provider", side_effect=ValueError("no key"))
    @patch("src.api.websocket.create_llm_provider")
    @patch("src.api.websocket.create_embedding_provider")
    @patch("src.api.websocket.get_qdrant")
    @patch("src.api.websocket.get_settings")
    def test_ws_streams_tokens(
        self, mock_settings, mock_get_qdrant, mock_embed, mock_llm_factory,
        mock_reranker, test_settings,
    ):
        """WebSocket streams tokens and sends sources/metadata/done."""
        mock_settings.return_value = test_settings

        # Mock qdrant
        mock_qdrant = AsyncMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.hybrid_search.return_value = [
            SearchResult(id="r1", score=0.9, payload={
                "chunk_text": "context", "document_id": "d1",
                "filename": "f.pdf", "page_numbers": [1],
                "header_chain": ["H1"], "parent_chunk_id": "", "chunk_index": 0,
            }),
        ]
        mock_get_qdrant.return_value = mock_qdrant

        # Mock embedding
        mock_embed_provider = AsyncMock()
        mock_embed_provider.embed_query.return_value = [0.1] * 1024
        mock_embed.return_value = mock_embed_provider

        # Mock LLM with streaming
        mock_llm = AsyncMock()
        mock_llm.model_name = "test-model"

        async def mock_stream(*args, **kwargs):
            for token in ["Hello", " world"]:
                yield token

        mock_llm.generate_stream = mock_stream
        mock_llm_factory.return_value = mock_llm

        client = TestClient(app)
        with client.websocket_connect("/ws/query") as ws:
            ws.send_json({
                "query": "test question",
                "collection_ids": ["col1"],
                "options": {"multi_query": False, "corrective_rag": False, "context_expansion": "off"},
            })

            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] in ("done", "error"):
                    break

            types = [m["type"] for m in messages]
            assert "token" in types
            assert "sources" in types
            assert "metadata" in types
            assert types[-1] == "done"

            # Check token content
            tokens = [m["content"] for m in messages if m["type"] == "token"]
            assert "Hello" in tokens
