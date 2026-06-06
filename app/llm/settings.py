"""Settings for the LLM (OpenAI) used in semantic extraction.

Values are read from environment variables (prefixed ``OPENAI_``) or a local
``.env`` file, mirroring :class:`config.IrisSettings`. ``OPENAI_API_KEY`` is the
SDK's conventional key name, so it is picked up here too.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Reasoning effort levels accepted by the Responses API for reasoning models.
# Higher effort improves accuracy on ambiguous questions at the cost of latency.
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class LLMSettings(BaseSettings):
    """OpenAI connection + model parameters."""

    api_key: str | None = None
    model: str = "gpt-5.4-nano"
    # The model's own default effort is "none"; bump it so extraction/binding
    # actually reason about ambiguous phrasing. Override via OPENAI_REASONING_EFFORT.
    reasoning_effort: ReasoningEffort = "medium"

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Return cached LLM settings so the environment is read only once."""
    return LLMSettings()
