"""Application configuration with Pydantic Settings.

Loads from environment variables with optional YAML config file override.
Environment variables always take precedence over YAML values.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ParserType(StrEnum):
    DOCLING = "docling"
    PYMUPDF4LLM = "pymupdf4llm"
    VISION = "vision"


class ContextExpansionMode(StrEnum):
    OFF = "off"
    PARENT = "parent"
    SIBLINGS = "siblings"


class Settings(BaseSettings):
    """Application settings loaded from env vars and optional config.yaml."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Embedding Provider ---
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 64

    # --- LLM Provider ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-5-20250929"

    # --- Reranker Provider ---
    reranker_provider: str = "cohere"
    reranker_model: str = "rerank-v3.5"

    # --- API Keys (loaded from env only) ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # --- Ingestion ---
    parser: ParserType = ParserType.DOCLING
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    contextual_enrichment: bool = True

    # --- Retrieval ---
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    hybrid_search: bool = True
    rrf_k: int = 60
    multi_query: bool = True
    corrective_rag: bool = True
    corrective_rag_threshold: float = 0.5
    context_expansion: ContextExpansionMode = ContextExpansionMode.PARENT
    max_context_tokens: int = 8000

    # --- Storage ---
    qdrant_url: str = "http://localhost:6333"
    database_url: str = Field(default="sqlite+aiosqlite:///data/metadata.db")

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    # --- Logging ---
    log_level: str = "INFO"

    # --- Upload ---
    max_upload_size_mb: int = 100
    upload_dir: str = "data/uploads"


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file if it exists."""
    if config_path is None:
        config_path = Path("config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    flat: dict[str, Any] = {}
    for section_key, section_val in data.items():
        if isinstance(section_val, dict):
            for key, val in section_val.items():
                flat_key = f"{key}" if section_key in ("providers",) else f"{section_key}_{key}"
                if isinstance(val, dict):
                    for nested_key, nested_val in val.items():
                        flat[f"{flat_key}_{nested_key}" if flat_key != key else nested_key] = (
                            nested_val
                        )
                else:
                    flat[flat_key] = val
        else:
            flat[section_key] = section_val
    return flat


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings. Env vars override YAML values."""
    yaml_config = load_yaml_config()
    return Settings(**yaml_config)
