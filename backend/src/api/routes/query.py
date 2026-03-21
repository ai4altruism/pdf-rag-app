"""Query API — submit questions and get RAG-powered answers."""

from fastapi import APIRouter, Depends

from src.config import Settings, get_settings
from src.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def submit_query(
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
):
    """Submit a query and get an answer with sources.

    Full implementation in Sprint 5-6. Currently returns a stub response.
    """
    return QueryResponse(
        answer="Query pipeline not yet implemented. See Sprint 5-6.",
        sources=[],
        metadata={
            "retrieval_count": 0,
            "reranked_count": 0,
            "corrective_rag_triggered": False,
            "query_variations": [body.query],
            "latency_ms": 0,
            "model_used": settings.llm_model,
        },
    )
