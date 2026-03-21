"""API layer — route registration."""

from fastapi import APIRouter

from src.api.routes import collections, conversations, documents, query, settings

api_router = APIRouter(prefix="/api")
api_router.include_router(documents.router)
api_router.include_router(collections.router)
api_router.include_router(query.router)
api_router.include_router(conversations.router)
api_router.include_router(settings.router)
