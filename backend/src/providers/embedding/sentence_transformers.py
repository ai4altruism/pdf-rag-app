"""Sentence-Transformers embedding provider for local model inference."""

import asyncio
from functools import partial

from src.logging import get_logger
from src.providers.base import EmbeddingProvider

logger = get_logger(__name__)


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers for local inference.

    Supports any HuggingFace model string. Runs on CPU by default,
    GPU when available.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        dimensions: int | None = None,
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        self._st_model = SentenceTransformer(model)
        self._model_name = model
        # Use model's native dimensions unless explicitly overridden
        self._dimensions = dimensions or self._st_model.get_sentence_embedding_dimension()
        self._batch_size = batch_size

        logger.info(
            "sentence_transformers_loaded",
            model=model,
            dimensions=self._dimensions,
            device=str(self._st_model.device),
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the local model."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            partial(self._encode, texts),
        )
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self.embed_texts([query])
        return result[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Synchronous encoding — runs in thread pool."""
        embeddings = self._st_model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]
