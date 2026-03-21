"""Settings API — get and update provider configuration."""

from fastapi import APIRouter, Depends

from src.config import Settings, get_settings
from src.schemas import SettingsResponse

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def get_current_settings(settings: Settings = Depends(get_settings)):
    """Get current provider and pipeline configuration."""
    return SettingsResponse(
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        reranker_provider=settings.reranker_provider,
        reranker_model=settings.reranker_model,
        parser=settings.parser,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        contextual_enrichment=settings.contextual_enrichment,
        top_k_retrieval=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
        hybrid_search=settings.hybrid_search,
        multi_query=settings.multi_query,
        corrective_rag=settings.corrective_rag,
        context_expansion=settings.context_expansion,
        max_context_tokens=settings.max_context_tokens,
    )


@router.put("", response_model=SettingsResponse)
async def update_settings(settings: Settings = Depends(get_settings)):
    """Update provider and pipeline configuration.

    Full implementation in Sprint 7 (persisted settings).
    Currently returns current settings as read-only.
    """
    return await get_current_settings(settings)
