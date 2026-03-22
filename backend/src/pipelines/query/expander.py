"""Multi-query expansion — generates alternative phrasings to improve retrieval recall."""

import json

from src.logging import get_logger
from src.providers.base import LLMProvider

logger = get_logger(__name__)

EXPANSION_PROMPT = """\
You are a query expansion assistant for a document retrieval system.
Given the user's original query, generate {count} alternative phrasings that:
- Preserve the original intent
- Use different vocabulary / synonyms
- Vary in specificity (one broader, one more specific)

Return ONLY a JSON array of strings. No explanation.

Original query: {query}"""


class QueryExpander:
    """Generates alternative query phrasings via LLM to improve retrieval recall."""

    def __init__(self, llm: LLMProvider, num_variations: int = 3):
        self._llm = llm
        self._num_variations = num_variations

    async def expand(self, query: str) -> list[str]:
        """Return the original query plus LLM-generated alternative phrasings.

        Always includes the original query as the first element.
        On failure, returns just the original query.
        """
        try:
            prompt = EXPANSION_PROMPT.format(query=query, count=self._num_variations)
            response = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            variations = self._parse_response(response)
            if not variations:
                logger.warning("expansion_empty", query=query)
                return [query]

            result = [query, *variations]
            logger.info(
                "query_expanded",
                original=query,
                variation_count=len(variations),
            )
            return result

        except Exception as e:
            logger.error("expansion_failed", query=query, error=str(e))
            return [query]

    def _parse_response(self, response: str) -> list[str]:
        """Parse LLM response into a list of query strings."""
        text = response.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            logger.warning("expansion_parse_failed", response=text[:200])
        return []
