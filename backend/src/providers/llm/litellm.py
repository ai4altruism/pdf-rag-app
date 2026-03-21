"""LLM provider using LiteLLM for unified access to Anthropic, OpenAI, Ollama, and others."""

import asyncio
from collections.abc import AsyncIterator

import litellm

from src.logging import get_logger
from src.providers.base import LLMProvider

logger = get_logger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


class LiteLLMProvider(LLMProvider):
    """LLM provider that routes to any supported backend via LiteLLM."""

    def __init__(self, model: str, api_key: str = ""):
        self._model = model
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate a complete response."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await litellm.acompletion(
                    model=self._model,
                    messages=messages,
                    api_key=self._api_key or None,
                    **kwargs,
                )
                return response.choices[0].message.content
            except litellm.RateLimitError:
                if attempt == MAX_RETRIES:
                    raise
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("rate_limited", provider="litellm", attempt=attempt)
                await asyncio.sleep(delay)
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "llm_error",
                    provider="litellm",
                    model=self._model,
                    error=str(e),
                    attempt=attempt,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable: retry loop exited without return or raise")

    async def generate_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Stream response tokens."""
        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            api_key=self._api_key or None,
            stream=True,
            **kwargs,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
