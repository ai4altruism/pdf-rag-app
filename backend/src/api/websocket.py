"""WebSocket endpoint for streaming query responses."""

import contextlib
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config import get_settings
from src.logging import get_logger
from src.pipelines.query.generator import AnswerGenerator
from src.pipelines.query.pipeline import (
    INSUFFICIENT_CONTEXT_MSG,
    QueryPipeline,
    QueryPipelineConfig,
)
from src.providers import create_embedding_provider, create_llm_provider, create_reranker_provider
from src.storage import get_qdrant

router = APIRouter()
logger = get_logger(__name__)


@router.websocket("/ws/query")
async def query_stream(websocket: WebSocket):
    """Stream query responses via WebSocket.

    Receives JSON messages with query parameters, streams answer tokens,
    then sends a final message with sources and metadata.

    Expected message format:
    {
        "query": "...",
        "collection_ids": ["..."],
        "options": { ... }  // optional, same as QueryOptions
    }

    Response messages:
    - {"type": "token", "content": "..."} — streaming answer tokens
    - {"type": "sources", "content": [...]} — source citations after stream
    - {"type": "metadata", "content": {...}} — pipeline metadata
    - {"type": "done"} — stream complete
    - {"type": "error", "content": "..."} — error occurred
    """
    await websocket.accept()
    logger.info("websocket_connected")

    try:
        while True:
            data = await websocket.receive_json()
            await _handle_query(websocket, data)
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "content": str(e)})


async def _handle_query(websocket: WebSocket, data: dict) -> None:
    """Process a single query message and stream the response."""
    query = data.get("query", "")
    collection_ids = data.get("collection_ids", [])
    options = data.get("options", {})

    if not query or not collection_ids:
        await websocket.send_json({
            "type": "error",
            "content": "Missing required fields: query, collection_ids",
        })
        return

    settings = get_settings()

    try:
        embedding_provider = create_embedding_provider(settings)
        try:
            llm_provider = create_llm_provider(settings)
        except (ValueError, Exception):
            llm_provider = None

        try:
            reranker_provider = create_reranker_provider(settings)
        except (ValueError, Exception):
            reranker_provider = None

        if not llm_provider:
            await websocket.send_json({
                "type": "error",
                "content": "LLM provider not configured. Streaming requires an LLM.",
            })
            return

        # Run retrieval pipeline
        pipeline = QueryPipeline(
            qdrant=get_qdrant(),
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            reranker_provider=reranker_provider,
        )

        config = QueryPipelineConfig(
            multi_query=options.get("multi_query", settings.multi_query),
            hybrid_search=settings.hybrid_search,
            rrf_k=settings.rrf_k,
            top_k_retrieval=options.get("top_k_retrieval", settings.top_k_retrieval),
            top_k_rerank=options.get("top_k_rerank", settings.top_k_rerank),
            context_expansion=options.get("context_expansion", settings.context_expansion),
            max_context_tokens=options.get("max_context_tokens", settings.max_context_tokens),
            embedding_model_filter=settings.embedding_model,
            corrective_rag=options.get("corrective_rag", settings.corrective_rag),
            corrective_rag_threshold=settings.corrective_rag_threshold,
        )

        result = await pipeline.run(query, collection_ids, config)

        # Handle insufficient context
        if result.insufficient_context or not result.context_chunks:
            await websocket.send_json({
                "type": "token",
                "content": INSUFFICIENT_CONTEXT_MSG,
            })
        else:
            # Stream answer tokens
            generator = AnswerGenerator(llm_provider)
            async for token in generator.generate_stream(query, result.context_chunks):
                await websocket.send_json({"type": "token", "content": token})

        # Send sources
        sources = [
            {
                "document_id": r.payload.get("document_id", ""),
                "filename": r.payload.get("filename", ""),
                "page_numbers": r.payload.get("page_numbers", []),
                "header_chain": r.payload.get("header_chain", []),
                "chunk_text": r.payload.get("chunk_text", ""),
                "relevance_score": round(r.score, 4),
            }
            for r in result.context_chunks
        ]
        await websocket.send_json({"type": "sources", "content": sources})

        # Send metadata
        await websocket.send_json({
            "type": "metadata",
            "content": {
                "retrieval_count": result.retrieval_count,
                "reranked_count": result.reranked_count,
                "corrective_rag_triggered": result.corrective_rag_triggered,
                "query_variations": result.query_variations,
                "latency_ms": result.latency_ms,
                "model_used": settings.llm_model,
            },
        })

        await websocket.send_json({"type": "done"})

    except Exception as e:
        logger.error("ws_query_error", error=str(e), traceback=traceback.format_exc())
        await websocket.send_json({"type": "error", "content": str(e)})
