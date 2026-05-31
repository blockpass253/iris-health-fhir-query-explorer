"""Persistence for the semantic registry (JSON on local disk).

The registry is the canonical runtime semantic layer. It is written to a single
file, overwritten on each (re)index.
"""

from pathlib import Path

from app.schema.models.registry import SchemaRegistry

DEFAULT_REGISTRY_PATH = Path("data/schema_registry.json")


def save_registry(registry: SchemaRegistry, path: Path = DEFAULT_REGISTRY_PATH) -> Path:
    """Write ``registry`` to ``path`` as indented JSON, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> SchemaRegistry:
    """Load a previously persisted registry from ``path``."""
    return SchemaRegistry.model_validate_json(path.read_text(encoding="utf-8"))
