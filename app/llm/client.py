"""Cached OpenAI async client factory.

The client is built from :func:`app.llm.settings.get_llm_settings`. When
``api_key`` is unset, the SDK still falls back to the ``OPENAI_API_KEY``
environment variable.
"""

from functools import lru_cache

from openai import AsyncOpenAI

from app.llm.settings import get_llm_settings


@lru_cache
def get_async_client() -> AsyncOpenAI:
    """Return a cached :class:`AsyncOpenAI` configured from settings."""
    settings = get_llm_settings()
    return AsyncOpenAI(api_key=settings.api_key)
