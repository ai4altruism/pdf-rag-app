"""Ollama embedding provider for local model inference via Ollama API."""

import httpx

from src.logging import get_logger
from src.providers.base import EmbeddingProvider

logger = get_logger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the Ollama REST API.

    Requires Ollama to be running locally with the configured model pulled.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._dimensions = dimensions
        self._base_url = base_url.rstrip("/")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via Ollama API."""
        all_embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            for text in texts:
                embedding = await self._embed_single(client, text)
                all_embeddings.append(embedding)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await self._embed_single(client, query)

    async def _embed_single(self, client: httpx.AsyncClient, text: str) -> list[float]:
        """Call Ollama's embedding endpoint for a single text."""
        response = await client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]
