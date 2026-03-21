"""SQLAlchemy async database setup and session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


def get_engine(database_url: str | None = None):
    """Create the async database engine."""
    url = database_url or get_settings().database_url
    return create_async_engine(url, echo=False)


def get_session_factory(database_url: str | None = None) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory."""
    engine = get_engine(database_url)
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(database_url: str | None = None) -> None:
    """Create all tables. Used for initial setup."""
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
