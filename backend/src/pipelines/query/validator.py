"""Corrective RAG — validates retrieved chunk relevance and retries if needed."""

import json

from src.logging import get_logger
from src.providers.base import LLMProvider
from src.storage.qdrant import SearchResult

logger = get_logger(__name__)

RELEVANCE_PROMPT = """\
You are a relevance judge. Given a query and a set of retrieved text chunks, \
score how relevant the chunks are to answering the query.

Return ONLY a JSON object: {{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>"}}

A score of 1.0 means the chunks directly and fully address the query.
A score of 0.0 means the chunks are completely unrelated.

Query: {query}

Retrieved chunks:
{chunks}"""

REFORMULATION_PROMPT = """\
The following query did not retrieve relevant results from the document collection. \
Reformulate it to improve retrieval. Try different vocabulary, be more specific, \
or broaden the scope.

Return ONLY the reformulated query as a plain string, no quotes or explanation.

Original query: {query}"""


class RelevanceValidator:
    """Validates whether retrieved chunks are relevant to the query.

    If relevance is below threshold, reformulates the query for a retry.
    """

    def __init__(
        self,
        llm: LLMProvider,
        threshold: float = 0.5,
        max_attempts: int = 2,
    ):
        self._llm = llm
        self._threshold = threshold
        self._max_attempts = max_attempts

    async def validate(
        self, query: str, results: list[SearchResult]
    ) -> tuple[float, bool]:
        """Score the relevance of retrieved results to the query.

        Returns:
            Tuple of (relevance_score, is_relevant).
        """
        if not results:
            return 0.0, False

        chunks_text = "\n\n".join(
            f"[{i+1}] {r.payload.get('chunk_text', '')}"
            for i, r in enumerate(results[:5])  # Only judge top 5
        )

        prompt = RELEVANCE_PROMPT.format(query=query, chunks=chunks_text)

        try:
            response = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            score = self._parse_score(response)
            is_relevant = score >= self._threshold

            logger.info(
                "relevance_validated",
                query=query[:100],
                score=score,
                is_relevant=is_relevant,
                threshold=self._threshold,
            )
            return score, is_relevant

        except Exception as e:
            logger.error("relevance_validation_failed", error=str(e))
            # On failure, assume relevant to avoid blocking the user
            return 1.0, True

    async def reformulate(self, query: str) -> str:
        """Reformulate the query for a retry attempt."""
        prompt = REFORMULATION_PROMPT.format(query=query)

        try:
            response = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            reformulated = response.strip().strip('"').strip("'")
            logger.info(
                "query_reformulated",
                original=query[:100],
                reformulated=reformulated[:100],
            )
            return reformulated

        except Exception as e:
            logger.error("reformulation_failed", error=str(e))
            return query

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def _parse_score(self, response: str) -> float:
        """Parse the relevance score from LLM response."""
        text = response.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "score" in parsed:
                return float(parsed["score"])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        logger.warning("relevance_parse_failed", response=text[:200])
        return 1.0  # Assume relevant on parse failure
