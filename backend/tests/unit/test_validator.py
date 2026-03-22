"""Tests for corrective RAG relevance validator."""

import json
from unittest.mock import AsyncMock

import pytest

from src.pipelines.query.validator import RelevanceValidator
from src.storage.qdrant import SearchResult


def _make_result(text: str = "chunk text") -> SearchResult:
    return SearchResult(
        id="r1", score=0.9,
        payload={"chunk_text": text, "document_id": "d1"},
    )


@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    mock.model_name = "test-model"
    return mock


class TestRelevanceValidator:
    async def test_validate_relevant(self, mock_llm):
        mock_llm.generate.return_value = json.dumps({"score": 0.8, "reasoning": "good match"})

        validator = RelevanceValidator(mock_llm, threshold=0.5)
        score, is_relevant = await validator.validate("test query", [_make_result()])

        assert score == 0.8
        assert is_relevant is True

    async def test_validate_not_relevant(self, mock_llm):
        mock_llm.generate.return_value = json.dumps({"score": 0.2, "reasoning": "poor match"})

        validator = RelevanceValidator(mock_llm, threshold=0.5)
        score, is_relevant = await validator.validate("test query", [_make_result()])

        assert score == 0.2
        assert is_relevant is False

    async def test_validate_empty_results(self, mock_llm):
        validator = RelevanceValidator(mock_llm)
        score, is_relevant = await validator.validate("query", [])

        assert score == 0.0
        assert is_relevant is False
        mock_llm.generate.assert_not_called()

    async def test_validate_llm_failure_assumes_relevant(self, mock_llm):
        mock_llm.generate.side_effect = RuntimeError("API error")

        validator = RelevanceValidator(mock_llm)
        score, is_relevant = await validator.validate("query", [_make_result()])

        assert score == 1.0
        assert is_relevant is True

    async def test_validate_parse_failure_assumes_relevant(self, mock_llm):
        mock_llm.generate.return_value = "invalid json"

        validator = RelevanceValidator(mock_llm)
        score, is_relevant = await validator.validate("query", [_make_result()])

        assert score == 1.0
        assert is_relevant is True

    async def test_validate_handles_code_fences(self, mock_llm):
        mock_llm.generate.return_value = '```json\n{"score": 0.6, "reasoning": "ok"}\n```'

        validator = RelevanceValidator(mock_llm, threshold=0.5)
        score, is_relevant = await validator.validate("query", [_make_result()])

        assert score == 0.6
        assert is_relevant is True

    async def test_reformulate_returns_new_query(self, mock_llm):
        mock_llm.generate.return_value = "What are the key concepts of machine learning?"

        validator = RelevanceValidator(mock_llm)
        result = await validator.reformulate("ML stuff")

        assert result == "What are the key concepts of machine learning?"

    async def test_reformulate_strips_quotes(self, mock_llm):
        mock_llm.generate.return_value = '"Reformulated query"'

        validator = RelevanceValidator(mock_llm)
        result = await validator.reformulate("query")

        assert result == "Reformulated query"

    async def test_reformulate_failure_returns_original(self, mock_llm):
        mock_llm.generate.side_effect = RuntimeError("fail")

        validator = RelevanceValidator(mock_llm)
        result = await validator.reformulate("original query")

        assert result == "original query"

    async def test_max_attempts(self, mock_llm):
        validator = RelevanceValidator(mock_llm, max_attempts=3)
        assert validator.max_attempts == 3

    async def test_validate_only_judges_top_5(self, mock_llm):
        mock_llm.generate.return_value = json.dumps({"score": 0.8, "reasoning": "ok"})
        results = [_make_result(f"chunk {i}") for i in range(10)]

        validator = RelevanceValidator(mock_llm)
        await validator.validate("query", results)

        prompt = mock_llm.generate.call_args[0][0][0]["content"]
        # Should only include [1] through [5]
        assert "[5]" in prompt
        assert "[6]" not in prompt
