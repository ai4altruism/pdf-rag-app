"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import api_router
from src.api.websocket import router as ws_router
from src.config import get_settings
from src.logging import RequestIDMiddleware, setup_logging
from src.storage.database import init_db


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
async def health_check():
    """Check the health of all dependencies."""
    from src.schemas import DependencyStatus, HealthResponse

    # TODO: Real dependency checks (Sprint 2)
    return HealthResponse(
        status="ok",
        qdrant=DependencyStatus(status="ok", message="Not yet connected"),
        embedding_provider=DependencyStatus(status="ok", message="Not yet configured"),
        llm_provider=DependencyStatus(status="ok", message="Not yet configured"),
    )
