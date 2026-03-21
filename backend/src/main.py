"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import api_router
from src.api.websocket import router as ws_router
from src.config import get_settings
from src.logging import RequestIDMiddleware, setup_logging
from src.schemas import DependencyStatus, HealthResponse
from src.storage import get_qdrant
from src.storage.database import init_db
from src.storage.qdrant import QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown events."""
    settings = get_settings()
    setup_logging(settings.log_level)
    await init_db()
    yield


app = FastAPI(
    title="PDF RAG API",
    description="Production-grade Retrieval-Augmented Generation for PDF documents",
    version="2.0.0",
    lifespan=lifespan,
)

# Middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

# Routes
app.include_router(api_router)
app.include_router(ws_router)


@app.get("/api/health", tags=["health"])
async def health_check(qdrant: QdrantStore = Depends(get_qdrant)):
    """Check the health of all dependencies."""
    # Qdrant check
    qdrant_ok = await qdrant.health_check()
    qdrant_status = DependencyStatus(
        status="ok" if qdrant_ok else "error",
        message="Connected" if qdrant_ok else "Unreachable",
    )

    # Embedding provider check
    try:
        from src.providers import create_embedding_provider

        provider = create_embedding_provider(get_settings())
        embed_status = DependencyStatus(
            status="ok",
            message=f"Model: {provider.model_name} ({provider.dimensions}d)",
        )
    except Exception as e:
        embed_status = DependencyStatus(status="error", message=str(e))

    # LLM provider check
    try:
        from src.providers import create_llm_provider

        provider = create_llm_provider(get_settings())
        llm_status = DependencyStatus(
            status="ok",
            message=f"Model: {provider.model_name}",
        )
    except Exception as e:
        llm_status = DependencyStatus(status="error", message=str(e))

    overall = "ok" if all(
        s.status == "ok" for s in [qdrant_status, embed_status, llm_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        embedding_provider=embed_status,
        llm_provider=llm_status,
    )
