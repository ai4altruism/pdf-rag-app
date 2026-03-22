"""Provider abstraction layer — factory functions for creating providers from config."""

from src.config import Settings
from src.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Instantiate the configured embedding provider."""
    match settings.embedding_provider:
        case "openai":
            from src.providers.embedding.openai import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                batch_size=settings.embedding_batch_size,
            )
        case "cohere":
            from src.providers.embedding.cohere import CohereEmbeddingProvider

            return CohereEmbeddingProvider(
                api_key=settings.cohere_api_key,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
        case "sentence-transformers":
            from src.providers.embedding.sentence_transformers import (
                SentenceTransformersEmbeddingProvider,
            )

            return SentenceTransformersEmbeddingProvider(
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
        case "ollama":
            from src.providers.embedding.ollama import OllamaEmbeddingProvider

            return OllamaEmbeddingProvider(
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
        case _:
            raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate the configured LLM provider."""
    match settings.llm_provider:
        case "anthropic" | "openai" | "ollama":
            from src.providers.llm.litellm import LiteLLMProvider

            return LiteLLMProvider(
                model=f"{settings.llm_provider}/{settings.llm_model}"
                if settings.llm_provider != "openai"
                else settings.llm_model,
                api_key=_get_llm_api_key(settings),
            )
        case _:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def create_reranker_provider(settings: Settings) -> RerankerProvider:
    """Instantiate the configured reranker provider."""
    match settings.reranker_provider:
        case "cohere":
            from src.providers.reranker.cohere import CohereRerankerProvider

            return CohereRerankerProvider(
                api_key=settings.cohere_api_key,
                model=settings.reranker_model,
            )
        case "cross-encoder":
            from src.providers.reranker.cross_encoder import CrossEncoderRerankerProvider

            return CrossEncoderRerankerProvider(
                model=settings.reranker_model,
            )
        case _:
            raise ValueError(f"Unknown reranker provider: {settings.reranker_provider}")


def _get_llm_api_key(settings: Settings) -> str:
    """Resolve the API key for the configured LLM provider."""
    match settings.llm_provider:
        case "anthropic":
            return settings.anthropic_api_key
        case "openai":
            return settings.openai_api_key
        case "ollama":
            return ""
        case _:
            return ""
