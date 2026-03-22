"""Parent-child context expansion and token-budget assembly."""

import tiktoken

from src.logging import get_logger
from src.storage.qdrant import QdrantStore, SearchResult

logger = get_logger(__name__)


class ContextAssembler:
    """Assembles retrieved chunks into context, optionally expanding to parents/siblings.

    Modes:
        off      — Use matched child chunks as-is.
        parent   — Replace children with their parent chunks (deduplicated).
        siblings — Include matched children plus adjacent chunks from the same parent.

    Enforces a token budget to stay within LLM context limits.
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        max_context_tokens: int = 8000,
        encoding_name: str = "cl100k_base",
    ):
        self._qdrant = qdrant
        self._max_tokens = max_context_tokens
        self._encoder = tiktoken.get_encoding(encoding_name)

    async def assemble(
        self,
        results: list[SearchResult],
        collection_name: str,
        mode: str = "off",
    ) -> list[SearchResult]:
        """Assemble context from search results with optional expansion.

        Args:
            results: Ranked search results from retrieval/reranking.
            collection_name: Qdrant collection to fetch parents/siblings from.
            mode: Expansion mode — "off", "parent", or "siblings".

        Returns:
            List of SearchResult within the token budget.
        """
        if not results:
            return []

        if mode == "parent":
            results = await self._expand_parents(results, collection_name)
        elif mode == "siblings":
            results = await self._expand_siblings(results, collection_name)

        return self._enforce_budget(results)

    async def _expand_parents(
        self, results: list[SearchResult], collection_name: str
    ) -> list[SearchResult]:
        """Replace child chunks with their deduplicated parent chunks."""
        parent_ids: dict[str, float] = {}  # parent_id -> best child score
        non_parent_results: list[SearchResult] = []

        for r in results:
            parent_id = r.payload.get("parent_chunk_id", "")
            if parent_id:
                if parent_id not in parent_ids or r.score > parent_ids[parent_id]:
                    parent_ids[parent_id] = r.score
            else:
                # No parent — keep as-is (might be a parent chunk itself)
                non_parent_results.append(r)

        if not parent_ids:
            return results

        # Fetch parent chunks from Qdrant
        parent_points = await self._qdrant.get_points_by_ids(
            collection_name, list(parent_ids.keys())
        )

        parent_results = []
        for point in parent_points:
            point_id = str(point.id)
            parent_results.append(
                SearchResult(
                    id=point_id,
                    score=parent_ids.get(point_id, 0.0),
                    payload=point.payload or {},
                )
            )

        # Sort by score descending, parents first then non-parents
        parent_results.sort(key=lambda r: r.score, reverse=True)
        combined = parent_results + non_parent_results

        logger.info(
            "context_expanded_parents",
            original_count=len(results),
            parent_count=len(parent_results),
            combined_count=len(combined),
        )
        return combined

    async def _expand_siblings(
        self, results: list[SearchResult], collection_name: str
    ) -> list[SearchResult]:
        """Include matched chunks plus adjacent chunks from the same parent."""
        seen_ids: set[str] = set()
        expanded: list[SearchResult] = []

        for r in results:
            if r.id in seen_ids:
                continue
            seen_ids.add(r.id)
            expanded.append(r)

            parent_id = r.payload.get("parent_chunk_id", "")
            if not parent_id:
                continue

            # Fetch sibling chunks (same parent_chunk_id)
            sibling_points = await self._qdrant.get_points_by_filter(
                collection_name,
                filters={"parent_chunk_id": parent_id},
            )

            chunk_index = r.payload.get("chunk_index", 0)
            for point in sibling_points:
                point_id = str(point.id)
                if point_id in seen_ids:
                    continue
                # Only include immediate neighbors
                sibling_index = (point.payload or {}).get("chunk_index", -1)
                if abs(sibling_index - chunk_index) <= 1:
                    seen_ids.add(point_id)
                    expanded.append(
                        SearchResult(
                            id=point_id,
                            score=r.score * 0.9,  # slightly lower score for siblings
                            payload=point.payload or {},
                        )
                    )

        logger.info(
            "context_expanded_siblings",
            original_count=len(results),
            expanded_count=len(expanded),
        )
        return expanded

    def _enforce_budget(self, results: list[SearchResult]) -> list[SearchResult]:
        """Trim results to fit within the token budget."""
        budgeted: list[SearchResult] = []
        total_tokens = 0

        for r in results:
            text = r.payload.get("chunk_text", "")
            tokens = len(self._encoder.encode(text)) if text else 0
            if total_tokens + tokens > self._max_tokens and budgeted:
                break
            budgeted.append(r)
            total_tokens += tokens

        if len(budgeted) < len(results):
            logger.info(
                "context_budget_enforced",
                total_results=len(results),
                kept=len(budgeted),
                total_tokens=total_tokens,
                budget=self._max_tokens,
            )

        return budgeted
