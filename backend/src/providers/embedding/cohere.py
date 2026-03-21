"""Cohere embedding provider supporting embed-v4 with Matryoshka dimensions."""

import asyncio

import cohere

from src.logging import get_logger
from src.providers.base import EmbeddingProvider

logger = get_logger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0


class CohereEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the Cohere Embed API."""

    def __init__(
        self,
        api_key: str,
        model: str = "embed-v4.0",
        dimensions: int = 1024,
        batch_size: int = 96,
    ):
        self._client = cohere.AsyncClientV2(api_key=api_key)
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
        """Embed a batch of texts for document storage."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = await self._embed_with_retry(batch, input_type="search_document")
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self._embed_with_retry([query], input_type="search_query")
        return result[0]

    async def _embed_with_retry(
        self, texts: list[str], input_type: str
    ) -> list[list[float]]:
        """Call the Cohere embed API with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.embed(
                    texts=texts,
                    model=self._model,
                    input_type=input_type,
                    embedding_types=["float"],
                    output_dimension=self._dimensions,
                )
                return [list(emb) for emb in response.embeddings.float_]
            except cohere.TooManyRequestsError:
                if attempt == MAX_RETRIES:
                    raise
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "rate_limited",
                    provider="cohere",
                    attempt=attempt,
                    retry_delay=delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "api_error",
                    provider="cohere",
                    error=str(e),
                    attempt=attempt,
                    retry_delay=delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable: retry loop exited without return or raise")
