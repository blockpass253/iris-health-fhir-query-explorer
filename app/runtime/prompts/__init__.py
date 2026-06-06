"""Load LLM system prompts from co-located markdown files."""

from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent


@cache
def load_prompt(name: str) -> str:
    """Return the text of ``<name>.md`` from this package (cached)."""
    return (_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
