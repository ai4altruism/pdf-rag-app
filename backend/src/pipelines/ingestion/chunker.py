"""Document-aware chunker with parent-child chunk relationships.

Two-pass strategy:
1. Split on Markdown headers (H1-H3) to create structural "parent" chunks.
2. Sub-split oversized parents into token-limited "child" chunks with overlap.

Each child references its parent via parent_chunk_id, enabling context expansion
during retrieval (Sprint 5).
"""

import re
import uuid
from dataclasses import dataclass, field

import tiktoken

from src.logging import get_logger

logger = get_logger(__name__)

# Header pattern: matches # Header, ## Header, ### Header
HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    chunk_id: str
    text: str
    header_chain: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0
    parent_chunk_id: str = ""
    is_parent: bool = False


class DocumentChunker:
    """Document-aware chunker producing parent and child chunks.

    Args:
        chunk_size_tokens: Maximum tokens per child chunk.
        chunk_overlap_tokens: Token overlap between consecutive child chunks.
        encoding_name: Tiktoken encoding to use for token counting.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        self._chunk_size = chunk_size_tokens
        self._chunk_overlap = chunk_overlap_tokens
        self._encoder = tiktoken.get_encoding(encoding_name)

    def chunk(
        self,
        markdown: str,
        page_numbers: list[int] | None = None,
    ) -> list[Chunk]:
        """Split markdown into parent and child chunks.

        Args:
            markdown: The full Markdown text from a parsed PDF.
            page_numbers: Optional list of page numbers for the entire document.
                          Used as default when page-level mapping isn't available.

        Returns:
            List of Chunk objects (parents first, then children for each parent).
        """
        if not markdown.strip():
            return []

        default_pages = page_numbers or []

        # Pass 1: Split on headers into structural sections (parents)
        sections = self._split_on_headers(markdown)

        all_chunks: list[Chunk] = []
        child_index = 0

        for section in sections:
            parent_id = str(uuid.uuid4())
            parent_tokens = self._count_tokens(section["text"])

            parent = Chunk(
                chunk_id=parent_id,
                text=section["text"],
                header_chain=section["header_chain"],
                page_numbers=default_pages,
                chunk_index=0,
                token_count=parent_tokens,
                is_parent=True,
            )
            all_chunks.append(parent)

            # Pass 2: Sub-split oversized parents into children
            if parent_tokens <= self._chunk_size:
                # Small enough to be its own child
                child = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=section["text"],
                    header_chain=section["header_chain"],
                    page_numbers=default_pages,
                    chunk_index=child_index,
                    token_count=parent_tokens,
                    parent_chunk_id=parent_id,
                )
                all_chunks.append(child)
                child_index += 1
            else:
                # Split into token-limited children with overlap
                child_texts = self._split_tokens(section["text"])
                for child_text in child_texts:
                    child_tokens = self._count_tokens(child_text)
                    child = Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=child_text,
                        header_chain=section["header_chain"],
                        page_numbers=default_pages,
                        chunk_index=child_index,
                        token_count=child_tokens,
                        parent_chunk_id=parent_id,
                    )
                    all_chunks.append(child)
                    child_index += 1

        parents = [c for c in all_chunks if c.is_parent]
        children = [c for c in all_chunks if not c.is_parent]
        logger.info(
            "chunking_complete",
            parent_count=len(parents),
            child_count=len(children),
            total_tokens=sum(c.token_count for c in children),
        )

        return all_chunks

    def _split_on_headers(self, markdown: str) -> list[dict]:
        """Split markdown into sections based on H1-H3 headers.

        Returns a list of dicts with 'text' and 'header_chain' keys.
        """
        sections: list[dict] = []
        # Track active headers at each level
        active_headers: dict[int, str] = {}

        # Find all header positions
        header_matches = list(HEADER_PATTERN.finditer(markdown))

        if not header_matches:
            # No headers — treat entire text as one section
            return [{"text": markdown.strip(), "header_chain": []}]

        # Handle text before first header
        first_pos = header_matches[0].start()
        if first_pos > 0:
            preamble = markdown[:first_pos].strip()
            if preamble:
                sections.append({"text": preamble, "header_chain": []})

        for i, match in enumerate(header_matches):
            level = len(match.group(1))  # Number of # characters
            header_text = match.group(2).strip()

            # Update active headers — clear deeper levels
            active_headers[level] = header_text
            for deeper in list(active_headers.keys()):
                if deeper > level:
                    del active_headers[deeper]

            # Extract section text (from this header to the next)
            start = match.start()
            end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(markdown)
            section_text = markdown[start:end].strip()

            # Build the header chain in order
            header_chain = [
                active_headers[lvl] for lvl in sorted(active_headers.keys())
            ]

            sections.append({
                "text": section_text,
                "header_chain": header_chain,
            })

        return sections

    def _split_tokens(self, text: str) -> list[str]:
        """Split text into token-limited chunks with overlap.

        Uses a sentence-aware splitting strategy: tries to break on sentence
        boundaries ('. ', '\\n') before falling back to token boundaries.
        """
        tokens = self._encoder.encode(text)

        if len(tokens) <= self._chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoder.decode(chunk_tokens)

            # Try to break at a sentence boundary if not at the end
            if end < len(tokens):
                chunk_text = self._break_at_sentence(chunk_text)
                # Recalculate actual tokens consumed
                actual_tokens = len(self._encoder.encode(chunk_text))
                start += max(actual_tokens - self._chunk_overlap, 1)
            else:
                start = end

            chunks.append(chunk_text.strip())

        return [c for c in chunks if c]

    def _break_at_sentence(self, text: str) -> str:
        """Try to break text at the last sentence boundary."""
        # Look for the last sentence-ending punctuation in the final 20% of text
        cutoff = int(len(text) * 0.8)
        search_region = text[cutoff:]

        # Try sentence endings
        for sep in [". ", ".\n", "\n\n", "\n"]:
            pos = search_region.rfind(sep)
            if pos != -1:
                return text[: cutoff + pos + len(sep)]

        return text

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self._encoder.encode(text))
