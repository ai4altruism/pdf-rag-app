"""Tests for SQLAlchemy models."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.models import Collection, Conversation, Document, IngestionJob, Message
from src.storage.database import Base


@pytest.fixture
async def db():
    """Create a fresh in-memory database for each test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


class TestCollectionModel:
    async def test_create_collection(self, db: AsyncSession):
        collection = Collection(name="Test Collection", description="A test")
        db.add(collection)
        await db.flush()
        assert collection.id is not None
        assert collection.name == "Test Collection"
        assert collection.created_at is not None

    async def test_collection_unique_name(self, db: AsyncSession):
        db.add(Collection(name="Unique"))
        await db.flush()
        db.add(Collection(name="Unique"))
        with pytest.raises(Exception):  # noqa: B017
            await db.flush()


class TestDocumentModel:
    async def test_create_document(self, db: AsyncSession):
        collection = Collection(name="Docs")
        db.add(collection)
        await db.flush()

        doc = Document(
            collection_id=collection.id,
            filename="test.pdf",
            file_size_bytes=1024,
            file_hash="abc123",
            status="pending",
        )
        db.add(doc)
        await db.flush()
        assert doc.id is not None
        assert doc.status == "pending"
        assert doc.collection_id == collection.id


class TestConversationModel:
    async def test_create_conversation(self, db: AsyncSession):
        collection = Collection(name="Chat Docs")
        db.add(collection)
        await db.flush()

        conv = Conversation(collection_id=collection.id, title="Test Chat")
        db.add(conv)
        await db.flush()
        assert conv.id is not None
        assert conv.title == "Test Chat"


class TestMessageModel:
    async def test_create_message(self, db: AsyncSession):
        collection = Collection(name="Msg Docs")
        db.add(collection)
        await db.flush()

        conv = Conversation(collection_id=collection.id)
        db.add(conv)
        await db.flush()

        msg = Message(conversation_id=conv.id, role="user", content="Hello")
        db.add(msg)
        await db.flush()
        assert msg.id is not None
        assert msg.role == "user"


class TestIngestionJobModel:
    async def test_create_job(self, db: AsyncSession):
        collection = Collection(name="Job Docs")
        db.add(collection)
        await db.flush()

        doc = Document(collection_id=collection.id, filename="test.pdf", status="pending")
        db.add(doc)
        await db.flush()

        job = IngestionJob(document_id=doc.id, parser="docling")
        db.add(job)
        await db.flush()
        assert job.id is not None
        assert job.status == "queued"
        assert job.parser == "docling"
