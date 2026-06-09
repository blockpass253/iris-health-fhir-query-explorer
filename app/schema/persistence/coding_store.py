"""Persistence for the generated coding concept dictionary."""

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from app.runtime.models import Coding

DEFAULT_CODING_DICT_PATH = Path("data/coding_dictionary.json")


class CodingRef(BaseModel):
    """A terminology code without its system URI (the system is the outer group key)."""

    code: str
    display: str | None = None


class CodingDictionary(BaseModel):
    schema_name: str
    generated_at: datetime
    systems: dict[str, dict[str, CodingRef]]

    def lookup(self, concept: str) -> list[Coding]:
        """Return all codes for ``concept`` across every terminology system.

        Tries an exact key match first; falls back to a whole-word substring
        match so generic drug names like "metformin" match full product display
        strings like "24 hr metformin hydrochloride 500 mg ...".
        """
        key = concept.strip().lower()
        results: list[Coding] = []
        pattern: re.Pattern[str] | None = None
        for system, entries in self.systems.items():
            ref = entries.get(key)
            if ref is not None:
                results.append(
                    Coding(system=system, code=ref.code, display=ref.display)
                )
            else:
                if pattern is None:
                    pattern = re.compile(r"\b" + re.escape(key) + r"\b")
                for entry_key, entry_ref in entries.items():
                    if pattern.search(entry_key):
                        results.append(
                            Coding(
                                system=system,
                                code=entry_ref.code,
                                display=entry_ref.display,
                            )
                        )
        return results

    def set_coding(self, concept: str, coding: Coding) -> None:
        """Insert or overwrite ``concept`` under ``coding.system``."""
        key = concept.strip().lower()
        self.systems.setdefault(coding.system, {})[key] = CodingRef(
            code=coding.code, display=coding.display
        )

    @property
    def concept_count(self) -> int:
        """Distinct concept keys across all systems (may count duplicates)."""
        return sum(len(entries) for entries in self.systems.values())


def save_coding_dictionary(
    dictionary: CodingDictionary,
    path: Path = DEFAULT_CODING_DICT_PATH,
) -> Path:
    """Write ``dictionary`` to ``path`` as indented JSON, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dictionary.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_coding_dictionary(
    path: Path = DEFAULT_CODING_DICT_PATH,
) -> CodingDictionary | None:
    """Load a previously persisted coding dictionary, or None if absent."""
    if not path.exists():
        return None
    return CodingDictionary.model_validate_json(path.read_text(encoding="utf-8"))
