"""Contextual enrichment — generates LLM summaries to improve chunk embeddings.

For each chunk, calls the LLM to produce a 1-3 sentence summary describing
the chunk's position and topic within the document. The summary is prepended
to the chunk text before embedding, improving retrieval quality.

Summaries are cached by content hash to avoid redundant LLM calls on
re-ingestion when content hasn't changed.
"""

import hashlib

from src.logging import get_logger
from src.providers.base import LLMProvider

logger = get_logger(__name__)

ENRICHMENT_PROMPT = """You are an expert at contextualizing document chunks for retrieval.

Given the document title and a chunk of text from that document, write a brief 1-3 sentence
contextual summary that describes:
1. What this chunk is about
2. Where it fits within the broader document

Your summary should help a search engine understand this chunk's context.
Respond with ONLY the summary, no preamble.

Document title: {title}
Section: {header_chain}

Chunk:
{chunk_text}"""


class ContextualEnricher:
    """Generates contextual summaries for chunks via LLM.

    Args:
        llm_provider: The LLM provider to use for generating summaries.
    """

    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider
        self._cache: dict[str, str] = {}

    async def enrich_chunks(
        self,
        chunks: list[dict],
        document_title: str = "",
    ) -> list[dict]:
        """Enrich a list of chunks with contextual summaries.

        Args:
            chunks: List of chunk dicts with at least 'text', 'header_chain' keys.
            document_title: Title of the source document.

        Returns:
            The same chunks with 'contextual_summary' and 'text' updated
            (summary prepended to text).
        """
        enriched = []
        for chunk in chunks:
            text = chunk["text"]
            content_hash = self._hash_content(text)

            # Check cache first
            if content_hash in self._cache:
                summary = self._cache[content_hash]
                logger.debug("enrichment_cache_hit", hash=content_hash[:12])
            else:
                summary = await self._generate_summary(
                    text,
                    document_title=document_title,
                    header_chain=chunk.get("header_chain", []),
                )
                self._cache[content_hash] = summary

            enriched_chunk = dict(chunk)
            enriched_chunk["contextual_summary"] = summary
            enriched_chunk["text"] = f"{summary}\n\n{text}" if summary else text
            enriched.append(enriched_chunk)

        logger.info(
            "enrichment_complete",
            chunk_count=len(enriched),
            cache_size=len(self._cache),
        )
        return enriched

    async def _generate_summary(
        self,
        chunk_text: str,
        document_title: str = "",
        header_chain: list[str] | None = None,
    ) -> str:
        """Generate a contextual summary for a single chunk."""
        header_str = " > ".join(header_chain) if header_chain else "N/A"
        prompt = ENRICHMENT_PROMPT.format(
            title=document_title or "Untitled",
            header_chain=header_str,
            chunk_text=chunk_text[:2000],  # Limit input to avoid token overflow
        )

        try:
            summary = await self._llm.generate(
                messages=[{"role": "user", "content": prompt}],
            )
            return summary.strip()
        except Exception as e:
            logger.warning(
                "enrichment_failed",
                error=str(e),
                chunk_preview=chunk_text[:50],
            )
            return ""

    def load_cache(self, cache: dict[str, str]) -> None:
        """Load a pre-existing cache (e.g. from DB)."""
        self._cache.update(cache)

    def get_cache(self) -> dict[str, str]:
        """Return the current cache for persistence."""
        return dict(self._cache)

    @staticmethod
    def _hash_content(text: str) -> str:
        """Generate a SHA-256 hash of chunk content."""
        return hashlib.sha256(text.encode()).hexdigest()
