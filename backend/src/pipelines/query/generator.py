"""Answer generator — assembles context and generates grounded LLM responses."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.logging import get_logger
from src.providers.base import LLMProvider
from src.storage.qdrant import SearchResult

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful research assistant. Answer the user's question based ONLY on the \
provided context chunks. Follow these rules strictly:

1. Only use information from the provided context to answer.
2. Cite your sources using [Source N] notation, where N corresponds to the chunk number.
3. If the context does not contain enough information to answer the question, say so \
clearly — do NOT make up or infer information beyond what is provided.
4. Be concise and precise. Prefer direct answers over lengthy explanations.
5. If the question asks about something not covered in the context, respond: \
"The provided documents do not contain sufficient information to answer this question."
"""


@dataclass
class SourceCitation:
    """A source citation attached to a generated answer."""

    index: int
    document_id: str
    filename: str
    page_numbers: list[int] = field(default_factory=list)
    header_chain: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    chunk_text: str = ""


@dataclass
class GenerationResult:
    """Result from the answer generator."""

    answer: str
    citations: list[SourceCitation] = field(default_factory=list)


class AnswerGenerator:
    """Generates grounded answers from retrieved context chunks.

    Assembles context into a prompt, calls the LLM, and returns an answer
    with source citations.
    """

    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str | None = None,
    ):
        self._llm = llm
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    async def generate(
        self,
        query: str,
        context_chunks: list[SearchResult],
        conversation_history: list[dict] | None = None,
    ) -> GenerationResult:
        """Generate an answer from context chunks.

        Args:
            query: The user's question.
            context_chunks: Ranked and assembled context chunks.
            conversation_history: Optional prior messages for multi-turn context.

        Returns:
            GenerationResult with answer text and source citations.
        """
        citations = self._build_citations(context_chunks)
        context_text = self._format_context(context_chunks)
        messages = self._build_messages(query, context_text, conversation_history)

        answer = await self._llm.generate(messages, temperature=0.2)

        logger.info(
            "answer_generated",
            query=query[:100],
            context_chunks=len(context_chunks),
            answer_length=len(answer),
        )

        return GenerationResult(answer=answer, citations=citations)

    async def generate_stream(
        self,
        query: str,
        context_chunks: list[SearchResult],
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream answer tokens from context chunks.

        Yields individual tokens as they are generated.
        """
        context_text = self._format_context(context_chunks)
        messages = self._build_messages(query, context_text, conversation_history)

        async for token in self._llm.generate_stream(messages, temperature=0.2):
            yield token

    def _build_messages(
        self,
        query: str,
        context_text: str,
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        """Build the message list for the LLM."""
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add context and query
        user_content = f"""Context chunks:

{context_text}

Question: {query}"""

        messages.append({"role": "user", "content": user_content})
        return messages

    def _format_context(self, chunks: list[SearchResult]) -> str:
        """Format context chunks into a numbered text block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            payload = chunk.payload
            filename = payload.get("filename", "unknown")
            pages = payload.get("page_numbers", [])
            headers = payload.get("header_chain", [])
            text = payload.get("chunk_text", "")

            header = f"[Source {i}] {filename}"
            if pages:
                header += f" (p. {', '.join(str(p) for p in pages)})"
            if headers:
                header += f" — {' > '.join(headers)}"

            parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(parts)

    def _build_citations(self, chunks: list[SearchResult]) -> list[SourceCitation]:
        """Build source citations from context chunks."""
        return [
            SourceCitation(
                index=i + 1,
                document_id=chunk.payload.get("document_id", ""),
                filename=chunk.payload.get("filename", ""),
                page_numbers=chunk.payload.get("page_numbers", []),
                header_chain=chunk.payload.get("header_chain", []),
                relevance_score=chunk.score,
                chunk_text=chunk.payload.get("chunk_text", ""),
            )
            for i, chunk in enumerate(chunks)
        ]
