"""Tests for multi-query expansion."""

import json
from unittest.mock import AsyncMock

import pytest

from src.pipelines.query.expander import QueryExpander


@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    mock.model_name = "test-model"
    return mock


class TestQueryExpander:
    async def test_expand_returns_original_plus_variations(self, mock_llm):
        variations = ["What is machine learning?", "Explain ML concepts"]
        mock_llm.generate.return_value = json.dumps(variations)

        expander = QueryExpander(mock_llm, num_variations=2)
        result = await expander.expand("What is ML?")

        assert result[0] == "What is ML?"
        assert result[1:] == variations
        assert len(result) == 3

    async def test_expand_calls_llm_with_correct_prompt(self, mock_llm):
        mock_llm.generate.return_value = '["alt query"]'

        expander = QueryExpander(mock_llm, num_variations=2)
        await expander.expand("test query")

        call_args = mock_llm.generate.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert "test query" in messages[0]["content"]
        assert "2" in messages[0]["content"]
        assert call_args[1]["temperature"] == 0.7

    async def test_expand_returns_original_on_llm_failure(self, mock_llm):
        mock_llm.generate.side_effect = RuntimeError("API error")

        expander = QueryExpander(mock_llm)
        result = await expander.expand("test query")

        assert result == ["test query"]

    async def test_expand_returns_original_on_empty_response(self, mock_llm):
        mock_llm.generate.return_value = "[]"

        expander = QueryExpander(mock_llm)
        result = await expander.expand("test query")

        assert result == ["test query"]

    async def test_expand_handles_markdown_code_fences(self, mock_llm):
        mock_llm.generate.return_value = '```json\n["rephrased query"]\n```'

        expander = QueryExpander(mock_llm)
        result = await expander.expand("original")

        assert result == ["original", "rephrased query"]

    async def test_expand_handles_invalid_json(self, mock_llm):
        mock_llm.generate.return_value = "not valid json at all"

        expander = QueryExpander(mock_llm)
        result = await expander.expand("original")

        assert result == ["original"]

    async def test_expand_filters_non_string_items(self, mock_llm):
        mock_llm.generate.return_value = '["valid", 123, "", "also valid"]'

        expander = QueryExpander(mock_llm)
        result = await expander.expand("original")

        assert result == ["original", "valid", "also valid"]

    async def test_expand_diverse_phrasings(self, mock_llm):
        """Verify the LLM is asked for diverse phrasings."""
        variations = [
            "How does photosynthesis work in plants?",
            "Photosynthesis mechanism explained",
            "What are the steps of photosynthesis?",
        ]
        mock_llm.generate.return_value = json.dumps(variations)

        expander = QueryExpander(mock_llm, num_variations=3)
        result = await expander.expand("Explain photosynthesis")

        assert len(result) == 4
        assert result[0] == "Explain photosynthesis"
        assert all(v != "Explain photosynthesis" for v in result[1:])
