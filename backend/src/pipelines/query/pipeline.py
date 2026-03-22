"""Query pipeline orchestrator — expand → search → rerank → validate → assemble context."""

import time
from dataclasses import dataclass, field

from src.api.routes.collections import _qdrant_collection_name
from src.logging import get_logger
from src.pipelines.query.context import ContextAssembler
from src.pipelines.query.expander import QueryExpander
from src.pipelines.query.validator import RelevanceValidator
from src.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider
from src.storage.qdrant import QdrantStore, SearchResult

logger = get_logger(__name__)

INSUFFICIENT_CONTEXT_MSG = (
    "The provided documents do not contain sufficient information to answer this question."
)


@dataclass
class QueryPipelineConfig:
    """Configuration for the query pipeline."""

    multi_query: bool = True
    hybrid_search: bool = True
    rrf_k: int = 60
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    context_expansion: str = "parent"
    max_context_tokens: int = 8000
    embedding_model_filter: str = ""
    corrective_rag: bool = True
    corrective_rag_threshold: float = 0.5


@dataclass
class QueryPipelineResult:
    """Result from the query pipeline."""

    context_chunks: list[SearchResult] = field(default_factory=list)
    query_variations: list[str] = field(default_factory=list)
    retrieval_count: int = 0
    reranked_count: int = 0
    latency_ms: int = 0
    corrective_rag_triggered: bool = False
    retrieval_attempts: int = 1
    insufficient_context: bool = False


class QueryPipeline:
    """Orchestrates the full query pipeline: expand → search → rerank → validate → assemble.

    Each stage is independently configurable/skippable.
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
    ):
        self._qdrant = qdrant
        self._embedding = embedding_provider
        self._llm = llm_provider
        self._reranker = reranker_provider

    async def run(
        self,
        query: str,
        collection_ids: list[str],
        config: QueryPipelineConfig | None = None,
        filters: dict | None = None,
    ) -> QueryPipelineResult:
        """Execute the full query pipeline.

        Args:
            query: The user's query string.
            collection_ids: List of collection IDs to search across.
            config: Pipeline configuration. Uses defaults if not provided.
            filters: Additional metadata filters for retrieval.
        """
        cfg = config or QueryPipelineConfig()
        start_time = time.time()

        # --- Stage 1: Query expansion ---
        if cfg.multi_query and self._llm:
            expander = QueryExpander(self._llm)
            query_variations = await expander.expand(query)
        else:
            query_variations = [query]

        # --- Stage 2 + 3: Search + Rerank (with corrective RAG retry loop) ---
        corrective_triggered = False
        retrieval_attempts = 0
        all_results: list[SearchResult] = []
        current_query = query

        validator = (
            RelevanceValidator(self._llm, threshold=cfg.corrective_rag_threshold)
            if cfg.corrective_rag and self._llm
            else None
        )
        max_attempts = validator.max_attempts if validator else 1

        for attempt in range(1, max_attempts + 1):
            retrieval_attempts = attempt

            # Search
            search_queries = query_variations if attempt == 1 else [current_query]
            all_results = await self._search_collections(
                queries=search_queries,
                collection_ids=collection_ids,
                config=cfg,
                filters=filters,
            )
            all_results = self._deduplicate(all_results)
            all_results.sort(key=lambda r: r.score, reverse=True)
            all_results = all_results[: cfg.top_k_retrieval]

            # Rerank
            if self._reranker and all_results:
                all_results = await self._rerank(current_query, all_results, cfg.top_k_rerank)

            # Validate (corrective RAG)
            if validator and all_results and attempt < max_attempts:
                score, is_relevant = await validator.validate(current_query, all_results)
                if is_relevant:
                    break
                # Not relevant — reformulate and retry
                corrective_triggered = True
                current_query = await validator.reformulate(current_query)
                logger.info(
                    "corrective_rag_retry",
                    attempt=attempt,
                    score=score,
                    reformulated_query=current_query[:100],
                )
            else:
                break

        retrieval_count = len(all_results)
        reranked_count = len(all_results)

        # Check if we exhausted attempts without finding relevant results
        insufficient_context = False
        if validator and all_results and corrective_triggered:
            _, final_relevant = await validator.validate(current_query, all_results)
            if not final_relevant:
                insufficient_context = True

        # --- Stage 4: Context assembly ---
        assembler = ContextAssembler(
            self._qdrant, max_context_tokens=cfg.max_context_tokens
        )
        primary_collection = _qdrant_collection_name(collection_ids[0])
        context_chunks = await assembler.assemble(
            all_results, primary_collection, mode=cfg.context_expansion
        )

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "query_pipeline_complete",
            query=query[:100],
            variations=len(query_variations),
            retrieved=retrieval_count,
            reranked=reranked_count,
            context_chunks=len(context_chunks),
            corrective_rag=corrective_triggered,
            retrieval_attempts=retrieval_attempts,
            latency_ms=latency_ms,
        )

        return QueryPipelineResult(
            context_chunks=context_chunks,
            query_variations=query_variations,
            retrieval_count=retrieval_count,
            reranked_count=reranked_count,
            latency_ms=latency_ms,
            corrective_rag_triggered=corrective_triggered,
            retrieval_attempts=retrieval_attempts,
            insufficient_context=insufficient_context,
        )

    async def _search_collections(
        self,
        queries: list[str],
        collection_ids: list[str],
        config: QueryPipelineConfig,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search across multiple collections with all query variations."""
        all_results: list[SearchResult] = []

        search_filters = dict(filters or {})
        if config.embedding_model_filter:
            search_filters["embedding_model"] = config.embedding_model_filter

        for query_text in queries:
            query_vector = await self._embedding.embed_query(query_text)

            for col_id in collection_ids:
                qdrant_name = _qdrant_collection_name(col_id)
                if not await self._qdrant.collection_exists(qdrant_name):
                    continue

                if config.hybrid_search:
                    results = await self._qdrant.hybrid_search(
                        collection_name=qdrant_name,
                        query_dense=query_vector,
                        query_text=query_text,
                        top_k=config.top_k_retrieval,
                        rrf_k=config.rrf_k,
                        filters=search_filters or None,
                    )
                else:
                    results = await self._qdrant.dense_search(
                        collection_name=qdrant_name,
                        query_dense=query_vector,
                        top_k=config.top_k_retrieval,
                        filters=search_filters or None,
                    )

                all_results.extend(results)

        return all_results

    async def _rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Rerank results using the configured reranker."""
        documents = [r.payload.get("chunk_text", "") for r in results]

        reranked = await self._reranker.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
        )

        reranked_results = []
        for rr in reranked:
            original = results[rr.index]
            reranked_results.append(
                SearchResult(
                    id=original.id,
                    score=rr.score,
                    payload=original.payload,
                )
            )

        return reranked_results

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Deduplicate results by ID, keeping the highest score."""
        seen: dict[str, SearchResult] = {}
        for r in results:
            if r.id not in seen or r.score > seen[r.id].score:
                seen[r.id] = r
        return list(seen.values())
