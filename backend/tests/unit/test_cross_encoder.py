"""Tests for cross-encoder reranker provider."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    """Mock sentence-transformers since it may not be installed in test env."""
    mock_module = ModuleType("sentence_transformers")
    mock_cross_encoder = MagicMock()
    mock_module.CrossEncoder = mock_cross_encoder
    with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
        yield mock_cross_encoder


class TestCrossEncoderRerankerProvider:
    def _make_provider(self, mock_sentence_transformers):
        from src.providers.reranker.cross_encoder import CrossEncoderRerankerProvider

        provider = CrossEncoderRerankerProvider(model="test-model")
        return provider

    async def test_rerank_returns_top_k(self, mock_sentence_transformers):
        provider = self._make_provider(mock_sentence_transformers)
        # Mock predict to return scores
        instance = mock_sentence_transformers.return_value
        instance.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3])

        results = await provider.rerank(
            query="test query",
            documents=["doc1", "doc2", "doc3", "doc4"],
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].text == "doc2"
        assert results[0].score == pytest.approx(0.9)
        assert results[0].index == 1
        assert results[1].text == "doc3"
        assert results[1].score == pytest.approx(0.5)

    async def test_rerank_empty_documents(self, mock_sentence_transformers):
        provider = self._make_provider(mock_sentence_transformers)

        results = await provider.rerank(query="test", documents=[], top_k=5)

        assert results == []

    async def test_rerank_passes_query_doc_pairs(self, mock_sentence_transformers):
        provider = self._make_provider(mock_sentence_transformers)
        instance = mock_sentence_transformers.return_value
        instance.predict.return_value = np.array([0.5, 0.8])

        await provider.rerank(query="what is AI?", documents=["doc A", "doc B"], top_k=2)

        pairs = instance.predict.call_args[0][0]
        assert pairs == [["what is AI?", "doc A"], ["what is AI?", "doc B"]]

    async def test_rerank_preserves_original_index(self, mock_sentence_transformers):
        provider = self._make_provider(mock_sentence_transformers)
        instance = mock_sentence_transformers.return_value
        instance.predict.return_value = np.array([0.1, 0.9, 0.5])

        results = await provider.rerank(
            query="q", documents=["a", "b", "c"], top_k=3
        )

        # Sorted by score desc: b(idx=1), c(idx=2), a(idx=0)
        assert results[0].index == 1
        assert results[1].index == 2
        assert results[2].index == 0

    async def test_model_name(self, mock_sentence_transformers):
        provider = self._make_provider(mock_sentence_transformers)
        assert provider.model_name == "test-model"

    async def test_rerank_top_k_larger_than_docs(self, mock_sentence_transformers):
        provider = self._make_provider(mock_sentence_transformers)
        instance = mock_sentence_transformers.return_value
        instance.predict.return_value = np.array([0.5, 0.8])

        results = await provider.rerank(
            query="q", documents=["a", "b"], top_k=10
        )

        assert len(results) == 2

    async def test_factory_creates_cross_encoder(self, mock_sentence_transformers):
        from src.config import Settings

        settings = Settings(reranker_provider="cross-encoder", reranker_model="test-model")

        from src.providers import create_reranker_provider

        provider = create_reranker_provider(settings)

        assert provider.model_name == "test-model"
