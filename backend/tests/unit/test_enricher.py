"""Tests for the contextual enrichment module."""

from unittest.mock import AsyncMock

import pytest

from src.pipelines.ingestion.enricher import ContextualEnricher


@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    mock.generate.return_value = "This chunk discusses the introduction section."
    return mock


@pytest.fixture
def enricher(mock_llm):
    return ContextualEnricher(mock_llm)


@pytest.fixture
def sample_chunks():
    return [
        {
            "text": "This is the first paragraph of the document.",
            "header_chain": ["Introduction"],
            "chunk_id": "chunk-1",
        },
        {
            "text": "Here we discuss the methodology used.",
            "header_chain": ["Introduction", "Methodology"],
            "chunk_id": "chunk-2",
        },
    ]


class TestContextualEnricher:
    """Tests for ContextualEnricher."""

    async def test_enrich_chunks_adds_summary(self, enricher, sample_chunks):
        """Enrichment prepends summary to chunk text."""
        result = await enricher.enrich_chunks(sample_chunks, document_title="Test Doc")

        assert len(result) == 2
        for chunk in result:
            assert chunk["contextual_summary"] == "This chunk discusses the introduction section."
            assert chunk["text"].startswith("This chunk discusses")

    async def test_enrich_preserves_original_fields(self, enricher, sample_chunks):
        """Enrichment doesn't remove existing chunk fields."""
        result = await enricher.enrich_chunks(sample_chunks)

        for orig, enriched in zip(sample_chunks, result, strict=True):
            assert enriched["chunk_id"] == orig["chunk_id"]
            assert enriched["header_chain"] == orig["header_chain"]

    async def test_llm_called_for_each_chunk(self, enricher, mock_llm, sample_chunks):
        """LLM is called once per unique chunk."""
        await enricher.enrich_chunks(sample_chunks)
        assert mock_llm.generate.call_count == 2

    async def test_cache_prevents_duplicate_calls(self, enricher, mock_llm):
        """Identical chunk text uses the cache instead of calling LLM again."""
        chunks = [
            {"text": "Same text", "header_chain": [], "chunk_id": "a"},
            {"text": "Same text", "header_chain": [], "chunk_id": "b"},
        ]
        await enricher.enrich_chunks(chunks)
        # Only one LLM call since both chunks have identical text
        assert mock_llm.generate.call_count == 1

    async def test_cache_load_and_get(self, enricher, mock_llm):
        """Cache can be loaded and retrieved."""
        from src.pipelines.ingestion.enricher import ContextualEnricher

        text_hash = ContextualEnricher._hash_content("cached text")
        enricher.load_cache({text_hash: "Cached summary"})

        chunks = [{"text": "cached text", "header_chain": [], "chunk_id": "c"}]
        result = await enricher.enrich_chunks(chunks)

        assert result[0]["contextual_summary"] == "Cached summary"
        mock_llm.generate.assert_not_called()

        cache = enricher.get_cache()
        assert text_hash in cache

    async def test_llm_failure_returns_empty_summary(self, mock_llm):
        """If the LLM call fails, an empty summary is used."""
        mock_llm.generate.side_effect = RuntimeError("API down")
        enricher = ContextualEnricher(mock_llm)

        chunks = [{"text": "Some text", "header_chain": [], "chunk_id": "d"}]
        result = await enricher.enrich_chunks(chunks)

        assert result[0]["contextual_summary"] == ""
        # Text is unchanged when summary is empty
        assert result[0]["text"] == "Some text"

    async def test_empty_chunks_list(self, enricher):
        """Enriching an empty list returns an empty list."""
        result = await enricher.enrich_chunks([])
        assert result == []

    async def test_prompt_includes_title_and_headers(self, mock_llm):
        """Verify the LLM prompt includes document title and header chain."""
        enricher = ContextualEnricher(mock_llm)
        chunks = [{"text": "Content", "header_chain": ["Ch1", "Sec1"], "chunk_id": "e"}]
        await enricher.enrich_chunks(chunks, document_title="My Document")

        call_args = mock_llm.generate.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        prompt_text = messages[0]["content"]
        assert "My Document" in prompt_text
        assert "Ch1 > Sec1" in prompt_text
