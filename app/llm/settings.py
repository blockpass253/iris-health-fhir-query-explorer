"""Settings for the LLM (OpenAI) used in semantic extraction.

Values are read from environment variables (prefixed ``OPENAI_``) or a local
``.env`` file, mirroring :class:`config.IrisSettings`. ``OPENAI_API_KEY`` is the
SDK's conventional key name, so it is picked up here too.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """OpenAI connection + model parameters."""

    api_key: str | None = None
    model: str = "gpt-5.4-nano"

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
