"""Tests for additional embedding providers (Cohere, SentenceTransformers, Ollama)."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.providers import create_embedding_provider

# --- Cohere Embedding Provider ---


class TestCohereEmbeddingProvider:
    """Tests for CohereEmbeddingProvider with mocked Cohere client."""

    @patch("src.providers.embedding.cohere.cohere")
    def test_properties(self, mock_cohere_mod):
        from src.providers.embedding.cohere import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="test-key", model="embed-v4.0", dimensions=1024)
        assert provider.model_name == "embed-v4.0"
        assert provider.dimensions == 1024

    @patch("src.providers.embedding.cohere.cohere")
    async def test_embed_texts(self, mock_cohere_mod):
        from src.providers.embedding.cohere import CohereEmbeddingProvider

        mock_client = AsyncMock()
        mock_cohere_mod.AsyncClientV2.return_value = mock_client

        mock_response = MagicMock()
        mock_response.embeddings.float_ = [[0.1] * 1024, [0.2] * 1024]
        mock_client.embed.return_value = mock_response

        provider = CohereEmbeddingProvider(api_key="test-key", dimensions=1024)
        result = await provider.embed_texts(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 1024
        mock_client.embed.assert_called_once()

        # Verify input_type is search_document for texts
        call_kwargs = mock_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_document"

    @patch("src.providers.embedding.cohere.cohere")
    async def test_embed_query(self, mock_cohere_mod):
        from src.providers.embedding.cohere import CohereEmbeddingProvider

        mock_client = AsyncMock()
        mock_cohere_mod.AsyncClientV2.return_value = mock_client

        mock_response = MagicMock()
        mock_response.embeddings.float_ = [[0.5] * 1024]
        mock_client.embed.return_value = mock_response

        provider = CohereEmbeddingProvider(api_key="test-key", dimensions=1024)
        result = await provider.embed_query("test query")

        assert len(result) == 1024
        call_kwargs = mock_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_query"

    @patch("src.providers.embedding.cohere.cohere")
    async def test_retry_on_rate_limit(self, mock_cohere_mod):
        from src.providers.embedding.cohere import CohereEmbeddingProvider

        mock_client = AsyncMock()
        mock_cohere_mod.AsyncClientV2.return_value = mock_client
        mock_cohere_mod.TooManyRequestsError = type("TooManyRequestsError", (Exception,), {})

        mock_response = MagicMock()
        mock_response.embeddings.float_ = [[0.1] * 1024]
        mock_client.embed.side_effect = [
            mock_cohere_mod.TooManyRequestsError(),
            mock_response,
        ]

        provider = CohereEmbeddingProvider(api_key="test-key", dimensions=1024)
        result = await provider.embed_texts(["test"])

        assert len(result) == 1
        assert mock_client.embed.call_count == 2

    def test_factory_creates_cohere(self):
        """Factory creates CohereEmbeddingProvider for 'cohere' config."""
        settings = Settings(
            embedding_provider="cohere",
            embedding_model="embed-v4.0",
            embedding_dimensions=1024,
            cohere_api_key="test-key",
        )
        provider = create_embedding_provider(settings)
        assert provider.model_name == "embed-v4.0"


# --- SentenceTransformers Embedding Provider ---


class TestSentenceTransformersEmbeddingProvider:
    """Tests for SentenceTransformersEmbeddingProvider with mocked model."""

    @pytest.fixture
    def mock_st(self):
        """Mock sentence_transformers module."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            yield mock_model, mock_st_module

    def test_properties(self, mock_st):
        _mock_model, _ = mock_st
        import importlib

        import src.providers.embedding.sentence_transformers as mod
        importlib.reload(mod)

        provider = mod.SentenceTransformersEmbeddingProvider(model="test-model")
        assert provider.model_name == "test-model"
        assert provider.dimensions == 768

    async def test_embed_texts(self, mock_st):
        mock_model, _ = mock_st
        import importlib

        import src.providers.embedding.sentence_transformers as mod
        importlib.reload(mod)

        provider = mod.SentenceTransformersEmbeddingProvider(model="test-model")
        result = await provider.embed_texts(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 768
        mock_model.encode.assert_called_once()

    async def test_embed_query(self, mock_st):
        import importlib

        import numpy as np

        mock_model, _ = mock_st
        mock_model.encode.return_value = np.array([[0.5] * 768])

        import src.providers.embedding.sentence_transformers as mod
        importlib.reload(mod)

        provider = mod.SentenceTransformersEmbeddingProvider(model="test-model")
        result = await provider.embed_query("test query")

        assert len(result) == 768


# --- Ollama Embedding Provider ---


class TestOllamaEmbeddingProvider:
    """Tests for OllamaEmbeddingProvider with mocked HTTP."""

    def test_properties(self):
        from src.providers.embedding.ollama import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(model="nomic-embed-text", dimensions=768)
        assert provider.model_name == "nomic-embed-text"
        assert provider.dimensions == 768

    @patch("src.providers.embedding.ollama.httpx.AsyncClient")
    async def test_embed_texts(self, mock_client_cls):
        from src.providers.embedding.ollama import OllamaEmbeddingProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        provider = OllamaEmbeddingProvider(model="nomic-embed-text", dimensions=768)
        result = await provider.embed_texts(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 768
        mock_client.post.assert_called_once()

    @patch("src.providers.embedding.ollama.httpx.AsyncClient")
    async def test_embed_query(self, mock_client_cls):
        from src.providers.embedding.ollama import OllamaEmbeddingProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.5] * 768]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        provider = OllamaEmbeddingProvider(model="nomic-embed-text", dimensions=768)
        result = await provider.embed_query("test")

        assert len(result) == 768

    @patch("src.providers.embedding.ollama.httpx.AsyncClient")
    async def test_api_error_propagates(self, mock_client_cls):
        import httpx

        from src.providers.embedding.ollama import OllamaEmbeddingProvider

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=MagicMock()
        )
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        provider = OllamaEmbeddingProvider(model="nomic-embed-text", dimensions=768)
        with pytest.raises(httpx.HTTPStatusError):
            await provider.embed_texts(["test"])

    def test_factory_creates_ollama(self):
        """Factory creates OllamaEmbeddingProvider for 'ollama' config."""
        settings = Settings(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
        )
        provider = create_embedding_provider(settings)
        assert provider.model_name == "nomic-embed-text"


# --- Factory tests ---


class TestEmbeddingProviderFactory:
    """Tests for the embedding provider factory with new providers."""

    def test_unknown_provider_raises(self):
        settings = Settings(embedding_provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(settings)
