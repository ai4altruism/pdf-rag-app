"""Shared test fixtures for the PDF RAG backend."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import Settings, get_settings
from src.main import app
from src.storage.database import Base, get_db


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh in-memory SQLite database for each test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    return Settings(
        qdrant_url="http://localhost:6333",
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=1024,
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5-20250929",
        reranker_provider="cohere",
        reranker_model="rerank-v3.5",
        log_level="DEBUG",
    )


@pytest.fixture
async def client(test_settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client with overridden dependencies.

    Each request gets its own session from a shared in-memory DB so that
    commits from one request are visible to subsequent requests.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def override_get_settings() -> Settings:
        return test_settings

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = override_get_settings

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.fixture
def mock_embedding_provider() -> AsyncMock:
    """Mock embedding provider for unit tests."""
    mock = AsyncMock()
    mock.embed_texts.return_value = [[0.1] * 1024]
    mock.embed_query.return_value = [0.1] * 1024
    mock.dimensions = 1024
    mock.model_name = "test-model"
    return mock


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Mock LLM provider for unit tests."""
    mock = AsyncMock()
    mock.generate.return_value = "Test response"

    async def mock_stream(*args, **kwargs):
        for token in ["Test ", "streaming ", "response"]:
            yield token

    mock.generate_stream = mock_stream
    return mock
