"""Abstract base classes for all providers.

Every LLM, embedding, and reranker integration implements these interfaces,
making providers swappable via configuration.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class RerankResult:
    """A single reranked document with its score."""

    index: int
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Used for document ingestion."""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query. May use a different prompt/instruction than embed_texts."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier string."""
        ...


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate a complete response from a list of messages."""
        ...

    @abstractmethod
    async def generate_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Stream response tokens from a list of messages."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier string."""
        ...


class RerankerProvider(ABC):
    """Abstract interface for reranking providers."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[RerankResult]:
        """Rerank documents by relevance to the query. Returns top_k results."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier string."""
        ...
