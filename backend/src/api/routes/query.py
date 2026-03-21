"""Query API — submit questions and get RAG-powered answers."""

import time

from fastapi import APIRouter, Depends

from src.api.routes.collections import _qdrant_collection_name
from src.config import Settings, get_settings
from src.providers import create_embedding_provider
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
    """Submit a query and retrieve relevant chunks via hybrid search.

    Currently performs retrieval only (no reranking or generation — Sprint 5-6).
    """
    start_time = time.time()

    # Embed the query
    embedding_provider = create_embedding_provider(settings)
    query_vector = await embedding_provider.embed_query(body.query)

    # Search across all requested collections
    all_results = []
    for col_id in body.collection_ids:
        qdrant_name = _qdrant_collection_name(col_id)
        if not await qdrant.collection_exists(qdrant_name):
            continue

        if settings.hybrid_search:
            results = await qdrant.hybrid_search(
                collection_name=qdrant_name,
                query_dense=query_vector,
                query_text=body.query,
                top_k=body.options.top_k_retrieval,
                rrf_k=settings.rrf_k,
                filters={"embedding_model": settings.embedding_model},
            )
        else:
            results = await qdrant.dense_search(
                collection_name=qdrant_name,
                query_dense=query_vector,
                top_k=body.options.top_k_retrieval,
                filters={"embedding_model": settings.embedding_model},
            )
        all_results.extend(results)

    # Sort by score and take top_k
    all_results.sort(key=lambda r: r.score, reverse=True)
    top_results = all_results[: body.options.top_k_retrieval]

    # Build source responses
    sources = [
        SourceResponse(
            document_id=r.payload.get("document_id", ""),
            filename=r.payload.get("filename", ""),
            page_numbers=r.payload.get("page_numbers", []),
            header_chain=r.payload.get("header_chain", []),
            chunk_text=r.payload.get("chunk_text", ""),
            relevance_score=round(r.score, 4),
        )
        for r in top_results
    ]

    latency_ms = int((time.time() - start_time) * 1000)

    return QueryResponse(
        answer="Retrieval complete. Reranking and generation not yet implemented (Sprint 5-6).",
        sources=sources,
        metadata={
            "retrieval_count": len(top_results),
            "reranked_count": 0,
            "corrective_rag_triggered": False,
            "query_variations": [body.query],
            "latency_ms": latency_ms,
            "model_used": settings.llm_model,
        },
    )
