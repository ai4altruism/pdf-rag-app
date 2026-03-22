"""Query API — submit questions and get RAG-powered answers."""

from fastapi import APIRouter, Depends

from src.config import Settings, get_settings
from src.pipelines.query.pipeline import QueryPipeline, QueryPipelineConfig
from src.providers import create_embedding_provider, create_llm_provider, create_reranker_provider
from src.schemas import QueryRequest, QueryResponse, SourceResponse
from src.storage import get_qdrant
from src.storage.qdrant import QdrantStore

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def submit_query(
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant),
):
    """Submit a query and retrieve relevant chunks via the full query pipeline.

    Pipeline stages: multi-query expansion → hybrid search → rerank → context assembly.
    """
    embedding_provider = create_embedding_provider(settings)

    # LLM and reranker are optional — pipeline degrades gracefully without them
    try:
        llm_provider = create_llm_provider(settings)
    except (ValueError, Exception):
        llm_provider = None

    try:
        reranker_provider = create_reranker_provider(settings)
    except (ValueError, Exception):
        reranker_provider = None

    pipeline = QueryPipeline(
        qdrant=qdrant,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        reranker_provider=reranker_provider,
    )

    config = QueryPipelineConfig(
        multi_query=body.options.multi_query,
        hybrid_search=settings.hybrid_search,
        rrf_k=settings.rrf_k,
        top_k_retrieval=body.options.top_k_retrieval,
        top_k_rerank=body.options.top_k_rerank,
        context_expansion=body.options.context_expansion,
        max_context_tokens=body.options.max_context_tokens,
        embedding_model_filter=settings.embedding_model,
    )

    result = await pipeline.run(
        query=body.query,
        collection_ids=body.collection_ids,
        config=config,
    )

    # Build source responses from context chunks
    sources = [
        SourceResponse(
            document_id=r.payload.get("document_id", ""),
            filename=r.payload.get("filename", ""),
            page_numbers=r.payload.get("page_numbers", []),
            header_chain=r.payload.get("header_chain", []),
            chunk_text=r.payload.get("chunk_text", ""),
            relevance_score=round(r.score, 4),
        )
        for r in result.context_chunks
    ]

    return QueryResponse(
        answer="Context retrieved. LLM generation not yet implemented (Sprint 6).",
        sources=sources,
        metadata={
            "retrieval_count": result.retrieval_count,
            "reranked_count": result.reranked_count,
            "corrective_rag_triggered": False,
            "query_variations": result.query_variations,
            "latency_ms": result.latency_ms,
            "model_used": settings.llm_model,
        },
    )
