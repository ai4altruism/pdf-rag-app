"""Cross-encoder reranker using sentence-transformers."""

import asyncio
from functools import partial

from src.logging import get_logger
from src.providers.base import RerankerProvider, RerankResult

logger = get_logger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderRerankerProvider(RerankerProvider):
    """Reranker using a local cross-encoder model via sentence-transformers."""

    def __init__(self, model: str = DEFAULT_MODEL):
        from sentence_transformers import CrossEncoder

        self._model_name = model
        self._encoder = CrossEncoder(model)
        logger.info("cross_encoder_loaded", model=model)

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[RerankResult]:
        """Rerank documents using cross-encoder scoring."""
        if not documents:
            return []

        # Cross-encoder expects list of (query, document) pairs
        pairs = [[query, doc] for doc in documents]

        # Run blocking model inference in a thread pool
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, partial(self._predict, pairs)
        )

        # Build results sorted by score descending
        scored = [
            RerankResult(index=i, text=doc, score=float(score))
            for i, (doc, score) in enumerate(zip(documents, scores, strict=True))
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        results = scored[:top_k]

        logger.info(
            "rerank_complete",
            model=self._model_name,
            input_count=len(documents),
            output_count=len(results),
            top_score=results[0].score if results else 0.0,
        )
        return results

    def _predict(self, pairs: list[list[str]]) -> list[float]:
        """Synchronous prediction — called in executor."""
        return self._encoder.predict(pairs).tolist()
