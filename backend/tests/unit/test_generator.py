"""Tests for answer generator."""

from unittest.mock import AsyncMock

import pytest

from src.pipelines.query.generator import AnswerGenerator
from src.storage.qdrant import SearchResult


def _make_chunk(id_: str, score: float, text: str = "chunk text",
                filename: str = "doc.pdf", pages: list[int] | None = None) -> SearchResult:
    return SearchResult(
        id=id_,
        score=score,
        payload={
            "chunk_text": text,
            "document_id": "d1",
            "filename": filename,
            "page_numbers": pages or [1],
            "header_chain": ["Section 1"],
        },
    )


@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    mock.model_name = "test-model"
    mock.generate.return_value = "Based on the context, the answer is 42. [Source 1]"
    return mock


class TestAnswerGenerator:
    async def test_generate_returns_answer_and_citations(self, mock_llm):
        gen = AnswerGenerator(mock_llm)
        chunks = [_make_chunk("c1", 0.9, "The answer is 42")]

        result = await gen.generate("What is the answer?", chunks)

        assert "42" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].index == 1
        assert result.citations[0].filename == "doc.pdf"

    async def test_generate_passes_system_prompt(self, mock_llm):
        gen = AnswerGenerator(mock_llm)
        await gen.generate("q", [_make_chunk("c1", 0.9)])

        messages = mock_llm.generate.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "ONLY" in messages[0]["content"]

    async def test_generate_custom_system_prompt(self, mock_llm):
        gen = AnswerGenerator(mock_llm, system_prompt="You are a summarizer.")
        await gen.generate("q", [_make_chunk("c1", 0.9)])

        messages = mock_llm.generate.call_args[0][0]
        assert messages[0]["content"] == "You are a summarizer."

    async def test_generate_includes_context_in_message(self, mock_llm):
        gen = AnswerGenerator(mock_llm)
        chunks = [
            _make_chunk("c1", 0.9, "First chunk"),
            _make_chunk("c2", 0.8, "Second chunk", filename="other.pdf", pages=[3, 4]),
        ]

        await gen.generate("question?", chunks)

        messages = mock_llm.generate.call_args[0][0]
        user_msg = messages[-1]["content"]
        assert "[Source 1]" in user_msg
        assert "[Source 2]" in user_msg
        assert "First chunk" in user_msg
        assert "Second chunk" in user_msg
        assert "other.pdf" in user_msg
        assert "question?" in user_msg

    async def test_generate_with_conversation_history(self, mock_llm):
        gen = AnswerGenerator(mock_llm)
        history = [
            {"role": "user", "content": "What is X?"},
            {"role": "assistant", "content": "X is Y."},
        ]

        await gen.generate("Follow up", [_make_chunk("c1", 0.9)], conversation_history=history)

        messages = mock_llm.generate.call_args[0][0]
        # system + 2 history + 1 user = 4
        assert len(messages) == 4
        assert messages[1]["content"] == "What is X?"
        assert messages[2]["content"] == "X is Y."

    async def test_citations_preserve_scores(self, mock_llm):
        gen = AnswerGenerator(mock_llm)
        chunks = [
            _make_chunk("c1", 0.95),
            _make_chunk("c2", 0.82),
        ]

        result = await gen.generate("q", chunks)

        assert result.citations[0].relevance_score == 0.95
        assert result.citations[1].relevance_score == 0.82
        assert result.citations[0].index == 1
        assert result.citations[1].index == 2

    async def test_generate_uses_low_temperature(self, mock_llm):
        gen = AnswerGenerator(mock_llm)
        await gen.generate("q", [_make_chunk("c1", 0.9)])

        kwargs = mock_llm.generate.call_args[1]
        assert kwargs["temperature"] == 0.2

    async def test_generate_stream(self, mock_llm):
        async def mock_stream(*args, **kwargs):
            for token in ["The ", "answer ", "is ", "42."]:
                yield token

        mock_llm.generate_stream = mock_stream

        gen = AnswerGenerator(mock_llm)
        tokens = []
        async for token in gen.generate_stream("q", [_make_chunk("c1", 0.9)]):
            tokens.append(token)

        assert tokens == ["The ", "answer ", "is ", "42."]

    async def test_generate_empty_chunks(self, mock_llm):
        mock_llm.generate.return_value = "No context provided."
        gen = AnswerGenerator(mock_llm)

        result = await gen.generate("q", [])

        assert result.citations == []
        assert result.answer == "No context provided."
