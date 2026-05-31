"""Connection settings for the InterSystems IRIS instance.

Values are read from environment variables (prefixed ``IRIS_``) or a local
``.env`` file. See ``.env.example`` for the available keys.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class IrisSettings(BaseSettings):
    """IRIS connection parameters."""

    host: str = "localhost"
    port: int = 1972
    namespace: str = "USER"
    username: str = "_SYSTEM"
    password: str = "SYS"

    model_config = SettingsConfigDict(
        env_prefix="IRIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> IrisSettings:
    """Return cached IRIS settings so the environment is read only once."""
    return IrisSettings()
