"""Query API — submit questions and get RAG-powered answers."""

import json

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import Settings, get_settings
from src.models.conversation import Conversation
from src.models.message import Message
from src.pipelines.query.generator import AnswerGenerator
from src.pipelines.query.pipeline import (
    INSUFFICIENT_CONTEXT_MSG,
    QueryPipeline,
    QueryPipelineConfig,
)
from src.providers import create_embedding_provider, create_llm_provider, create_reranker_provider
from src.schemas import QueryRequest, QueryResponse, SourceResponse
from src.storage import get_qdrant
from src.storage.database import get_db
from src.storage.qdrant import QdrantStore

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def submit_query(
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant),
    db: AsyncSession = Depends(get_db),
):
    """Submit a query and get a RAG-powered answer.

    Pipeline: multi-query expansion → hybrid search → rerank → corrective RAG → generate.
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
        corrective_rag=body.options.corrective_rag,
        corrective_rag_threshold=settings.corrective_rag_threshold,
    )

    # Load conversation history if provided
    conversation_history = None
    if body.conversation_id:
        conversation_history = await _load_conversation_history(db, body.conversation_id)

    result = await pipeline.run(
        query=body.query,
        collection_ids=body.collection_ids,
        config=config,
    )

    # Generate answer
    if result.insufficient_context:
        answer = INSUFFICIENT_CONTEXT_MSG
    elif llm_provider and result.context_chunks:
        generator = AnswerGenerator(llm_provider)
        gen_result = await generator.generate(
            body.query, result.context_chunks, conversation_history
        )
        answer = gen_result.answer
    else:
        answer = "Context retrieved. LLM not available for generation."

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
        for r in result.context_chunks
    ]

    # Save messages to conversation if conversation_id provided
    if body.conversation_id:
        await _save_messages(
            db, body.conversation_id, body.query, answer, sources
        )

    return QueryResponse(
        answer=answer,
        sources=sources,
        metadata={
            "retrieval_count": result.retrieval_count,
            "reranked_count": result.reranked_count,
            "corrective_rag_triggered": result.corrective_rag_triggered,
            "query_variations": result.query_variations,
            "latency_ms": result.latency_ms,
            "model_used": settings.llm_model,
        },
    )


async def _load_conversation_history(
    db: AsyncSession, conversation_id: str
) -> list[dict] | None:
    """Load prior messages from a conversation for multi-turn context."""
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    result = await db.execute(stmt)
    conversation = result.scalar_one_or_none()
    if not conversation or not conversation.messages:
        return None

    history = []
    for msg in sorted(conversation.messages, key=lambda m: m.created_at):
        history.append({"role": msg.role, "content": msg.content})
    return history


async def _save_messages(
    db: AsyncSession,
    conversation_id: str,
    query: str,
    answer: str,
    sources: list[SourceResponse],
) -> None:
    """Save user query and assistant response to the conversation."""
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=query,
    )
    sources_json = json.dumps([s.model_dump() for s in sources])
    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=sources_json,
    )
    db.add(user_msg)
    db.add(assistant_msg)
    await db.flush()
