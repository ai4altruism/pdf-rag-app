"""Tests for the configuration system."""

import os
from unittest.mock import patch

from src.config import ContextExpansionMode, ParserType, Settings


class TestSettings:
    """Test Settings defaults and overrides."""

    def test_default_values(self):
        """Settings should have sensible defaults."""
        settings = Settings()
        assert settings.embedding_provider == "openai"
        assert settings.embedding_model == "text-embedding-3-large"
        assert settings.embedding_dimensions == 1024
        assert settings.llm_provider == "anthropic"
        assert settings.llm_model == "claude-sonnet-4-5-20250929"
        assert settings.reranker_provider == "cohere"
        assert settings.reranker_model == "rerank-v3.5"
        assert settings.parser == ParserType.DOCLING
        assert settings.chunk_size_tokens == 512
        assert settings.chunk_overlap_tokens == 50
        assert settings.contextual_enrichment is True
        assert settings.top_k_retrieval == 20
        assert settings.top_k_rerank == 5
        assert settings.hybrid_search is True
        assert settings.multi_query is True
        assert settings.corrective_rag is True
        assert settings.context_expansion == ContextExpansionMode.PARENT
        assert settings.max_context_tokens == 8000
        assert settings.max_upload_size_mb == 100
        assert settings.log_level == "INFO"

    def test_env_var_override(self):
        """Environment variables should override defaults."""
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "text-embedding-3-small"}):
            settings = Settings()
            assert settings.embedding_model == "text-embedding-3-small"

    def test_explicit_kwargs(self):
        """Explicit keyword arguments should work."""
        settings = Settings(
            embedding_provider="cohere",
            embedding_dimensions=512,
            llm_provider="openai",
            parser=ParserType.PYMUPDF4LLM,
        )
        assert settings.embedding_provider == "cohere"
        assert settings.embedding_dimensions == 512
        assert settings.llm_provider == "openai"
        assert settings.parser == ParserType.PYMUPDF4LLM

    def test_cors_origins_default(self):
        """CORS origins should default to localhost:3000."""
        settings = Settings()
        assert "http://localhost:3000" in settings.cors_origins
