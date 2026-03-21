"""Tests for provider interfaces and factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.providers import create_embedding_provider, create_llm_provider, create_reranker_provider
from src.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider, RerankResult


class TestProviderInterfaces:
    """Test that provider ABCs enforce the contract."""

    def test_embedding_provider_is_abstract(self):
        """EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_llm_provider_is_abstract(self):
        """LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_reranker_provider_is_abstract(self):
        """RerankerProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RerankerProvider()

    def test_rerank_result_dataclass(self):
        """RerankResult should hold index, text, score, and metadata."""
        result = RerankResult(index=0, text="hello", score=0.95)
        assert result.index == 0
        assert result.text == "hello"
        assert result.score == 0.95
        assert result.metadata == {}

    def test_rerank_result_with_metadata(self):
        result = RerankResult(index=1, text="world", score=0.8, metadata={"key": "val"})
        assert result.metadata == {"key": "val"}


class TestProviderFactory:
    """Test provider factory functions."""

    def test_create_embedding_provider_openai(self):
        """Factory should create OpenAI embedding provider."""
        settings = Settings(openai_api_key="test-key")
        provider = create_embedding_provider(settings)
        assert isinstance(provider, EmbeddingProvider)
        assert provider.model_name == "text-embedding-3-large"
        assert provider.dimensions == 1024

    def test_create_embedding_provider_unknown(self):
        """Factory should raise for unknown provider."""
        settings = Settings(embedding_provider="unknown")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(settings)

    def test_create_llm_provider_anthropic(self):
        """Factory should create LiteLLM provider for Anthropic."""
        settings = Settings(anthropic_api_key="test-key")
        provider = create_llm_provider(settings)
        assert isinstance(provider, LLMProvider)
        assert "anthropic" in provider.model_name

    def test_create_llm_provider_unknown(self):
        """Factory should raise for unknown LLM provider."""
        settings = Settings(llm_provider="unknown")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(settings)

    def test_create_reranker_provider_cohere(self):
        """Factory should create Cohere reranker provider."""
        settings = Settings(cohere_api_key="test-key")
        provider = create_reranker_provider(settings)
        assert isinstance(provider, RerankerProvider)
        assert provider.model_name == "rerank-v3.5"

    def test_create_reranker_provider_unknown(self):
        """Factory should raise for unknown reranker provider."""
        settings = Settings(reranker_provider="unknown")
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            create_reranker_provider(settings)


class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider with mocked API."""

    @pytest.fixture
    def provider(self):
        from src.providers.embedding.openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-large",
            dimensions=1024,
            batch_size=2,
        )

    async def test_embed_query(self, provider):
        """embed_query should return a single vector."""
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1] * 1024
        mock_response.data = [mock_item]

        with patch.object(
            provider._client.embeddings, "create", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await provider.embed_query("test query")
            assert len(result) == 1024

    async def test_embed_texts_batching(self, provider):
        """embed_texts should split into batches of batch_size."""
        mock_item = MagicMock()
        mock_item.embedding = [0.1] * 1024

        # First call gets 2 texts, second call gets 1 text
        resp_batch1 = MagicMock()
        resp_batch1.data = [mock_item, mock_item]
        resp_batch2 = MagicMock()
        resp_batch2.data = [mock_item]

        with patch.object(
            provider._client.embeddings,
            "create",
            new_callable=AsyncMock,
            side_effect=[resp_batch1, resp_batch2],
        ) as mock_create:
            result = await provider.embed_texts(["text1", "text2", "text3"])
            # batch_size=2, so 3 texts should produce 2 API calls
            assert mock_create.call_count == 2
            assert len(result) == 3

    async def test_properties(self, provider):
        assert provider.dimensions == 1024
        assert provider.model_name == "text-embedding-3-large"
