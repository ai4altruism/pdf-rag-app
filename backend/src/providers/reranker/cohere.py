"""Cohere Rerank provider using Rerank 3.5."""

import cohere

from src.logging import get_logger
from src.providers.base import RerankerProvider, RerankResult

logger = get_logger(__name__)


class CohereRerankerProvider(RerankerProvider):
    """Reranker using Cohere's Rerank API."""

    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[RerankResult]:
        """Rerank documents using Cohere Rerank API."""
        if not documents:
            return []

        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=True,
        )

        results = []
        for item in response.results:
            results.append(
                RerankResult(
                    index=item.index,
                    text=documents[item.index],
                    score=item.relevance_score,
                )
            )

        logger.info(
            "rerank_complete",
            model=self._model,
            input_count=len(documents),
            output_count=len(results),
            top_score=results[0].score if results else 0.0,
        )
        return results
