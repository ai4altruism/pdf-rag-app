"""OpenAI embedding provider supporting text-embedding-3-small/large with Matryoshka dimensions."""

import asyncio

import openai

from src.logging import get_logger
from src.providers.base import EmbeddingProvider

logger = get_logger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 1024,
        batch_size: int = 64,
    ):
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, splitting into sub-batches as needed."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = await self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self._embed_with_retry([query])
        return result[0]

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.embeddings.create(
                    input=texts,
                    model=self._model,
                    dimensions=self._dimensions,
                )
                return [item.embedding for item in response.data]
            except openai.RateLimitError:
                if attempt == MAX_RETRIES:
                    raise
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "rate_limited",
                    provider="openai",
                    attempt=attempt,
                    retry_delay=delay,
                )
                await asyncio.sleep(delay)
            except openai.APIError as e:
                if attempt == MAX_RETRIES:
                    raise
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "api_error",
                    provider="openai",
                    error=str(e),
                    attempt=attempt,
                    retry_delay=delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable: retry loop exited without return or raise")
