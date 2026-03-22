"""Tests for QueryPipeline orchestrator — covers S5-05 and S5-06."""

import json
from unittest.mock import AsyncMock

import pytest

from src.pipelines.query.pipeline import QueryPipeline, QueryPipelineConfig
from src.providers.base import RerankResult
from src.storage.qdrant import SearchResult


@pytest.fixture
def mock_qdrant():
    mock = AsyncMock()
    mock.collection_exists.return_value = True
    mock.hybrid_search.return_value = [
        SearchResult(id="r1", score=0.9, payload={
            "chunk_text": "result 1", "document_id": "d1", "filename": "f.pdf",
            "page_numbers": [1], "header_chain": ["H1"], "parent_chunk_id": "",
            "chunk_index": 0,
        }),
        SearchResult(id="r2", score=0.8, payload={
            "chunk_text": "result 2", "document_id": "d1", "filename": "f.pdf",
            "page_numbers": [2], "header_chain": ["H1"], "parent_chunk_id": "",
            "chunk_index": 1,
        }),
    ]
    mock.dense_search.return_value = [
        SearchResult(id="r1", score=0.85, payload={
            "chunk_text": "result 1", "document_id": "d1", "filename": "f.pdf",
            "page_numbers": [1], "header_chain": ["H1"], "parent_chunk_id": "",
            "chunk_index": 0,
        }),
    ]
    return mock


@pytest.fixture
def mock_embedding():
    mock = AsyncMock()
    mock.embed_query.return_value = [0.1] * 1024
    mock.model_name = "test-embed"
    return mock


@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    mock.model_name = "test-llm"
    mock.generate.return_value = json.dumps(["alt query 1", "alt query 2"])
    return mock


@pytest.fixture
def mock_reranker():
    mock = AsyncMock()
    mock.model_name = "test-reranker"
    mock.rerank.return_value = [
        RerankResult(index=0, text="result 1", score=0.95),
    ]
    return mock


class TestQueryPipelineBasic:
    async def test_run_full_pipeline(self, mock_qdrant, mock_embedding, mock_llm, mock_reranker):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
            llm_provider=mock_llm,
            reranker_provider=mock_reranker,
        )
        config = QueryPipelineConfig(
            multi_query=True,
            context_expansion="off",
            top_k_retrieval=20,
            top_k_rerank=5,
        )

        result = await pipeline.run("test query", ["col1"], config)

        assert result.retrieval_count > 0
        assert result.reranked_count > 0
        assert len(result.query_variations) == 3  # original + 2 expansions
        assert result.latency_ms >= 0

    async def test_run_without_multi_query(self, mock_qdrant, mock_embedding):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
        )
        config = QueryPipelineConfig(multi_query=False, context_expansion="off")

        result = await pipeline.run("test query", ["col1"], config)

        assert result.query_variations == ["test query"]
        mock_embedding.embed_query.assert_called_once_with("test query")

    async def test_run_without_reranker(self, mock_qdrant, mock_embedding):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
        )
        config = QueryPipelineConfig(multi_query=False, context_expansion="off")

        result = await pipeline.run("test query", ["col1"], config)

        # Without reranker, all retrieved results are kept
        assert result.retrieval_count == result.reranked_count

    async def test_run_dense_only(self, mock_qdrant, mock_embedding):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
        )
        config = QueryPipelineConfig(
            multi_query=False,
            hybrid_search=False,
            context_expansion="off",
        )

        result = await pipeline.run("test query", ["col1"], config)

        mock_qdrant.dense_search.assert_called_once()
        mock_qdrant.hybrid_search.assert_not_called()
        assert result.retrieval_count == 1


class TestMultiCollectionSearch:
    async def test_searches_multiple_collections(self, mock_qdrant, mock_embedding):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
        )
        config = QueryPipelineConfig(multi_query=False, context_expansion="off")

        result = await pipeline.run("query", ["col1", "col2", "col3"], config)

        # Should search each collection
        assert mock_qdrant.hybrid_search.call_count == 3
        # Results merged from all collections
        assert result.retrieval_count > 0

    async def test_skips_nonexistent_collections(self, mock_qdrant, mock_embedding):
        mock_qdrant.collection_exists.side_effect = lambda name: name == "col_col1"

        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
        )
        config = QueryPipelineConfig(multi_query=False, context_expansion="off")

        await pipeline.run("query", ["col1", "col2"], config)

        assert mock_qdrant.hybrid_search.call_count == 1

    async def test_deduplicates_across_collections(self, mock_qdrant, mock_embedding):
        # Same result from two collections
        mock_qdrant.hybrid_search.return_value = [
            SearchResult(id="same_id", score=0.9, payload={
                "chunk_text": "text", "parent_chunk_id": "", "chunk_index": 0,
            }),
        ]

        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
        )
        config = QueryPipelineConfig(multi_query=False, context_expansion="off")

        result = await pipeline.run("query", ["col1", "col2"], config)

        # Deduplicated
        assert result.retrieval_count == 1

    async def test_embedding_model_filter(self, mock_qdrant, mock_embedding):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
        )
        config = QueryPipelineConfig(
            multi_query=False,
            context_expansion="off",
            embedding_model_filter="text-embedding-3-large",
        )

        await pipeline.run("query", ["col1"], config)

        call_kwargs = mock_qdrant.hybrid_search.call_args[1]
        assert call_kwargs["filters"]["embedding_model"] == "text-embedding-3-large"

    async def test_additional_filters_passed(self, mock_qdrant, mock_embedding):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
        )
        config = QueryPipelineConfig(multi_query=False, context_expansion="off")

        await pipeline.run("query", ["col1"], config, filters={"document_id": "doc123"})

        call_kwargs = mock_qdrant.hybrid_search.call_args[1]
        assert call_kwargs["filters"]["document_id"] == "doc123"


class TestReranking:
    async def test_rerank_maps_scores_to_results(self, mock_qdrant, mock_embedding, mock_reranker):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
            reranker_provider=mock_reranker,
        )
        config = QueryPipelineConfig(
            multi_query=False,
            context_expansion="off",
            top_k_rerank=1,
        )

        result = await pipeline.run("query", ["col1"], config)

        assert result.reranked_count == 1
        # Reranked score should be from reranker, not original
        assert result.context_chunks[0].score == 0.95

    async def test_rerank_called_with_chunk_texts(self, mock_qdrant, mock_embedding, mock_reranker):
        pipeline = QueryPipeline(
            qdrant=mock_qdrant,
            embedding_provider=mock_embedding,
            reranker_provider=mock_reranker,
        )
        config = QueryPipelineConfig(multi_query=False, context_expansion="off")

        await pipeline.run("query", ["col1"], config)

        call_args = mock_reranker.rerank.call_args
        assert call_args[1]["query"] == "query"
        assert "result 1" in call_args[1]["documents"]
