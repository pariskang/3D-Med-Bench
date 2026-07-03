"""
Scripted Findings Library — non-simulatable findings stored as keyed objects.
Loaded from resources/findings_lib/findings.yaml.
Each finding has: result text, LR pairs, optional time trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from clinicraft.config import settings


class FindingEntry(BaseModel):
    finding_id: str
    display: str
    result_template: str          # may include {value} placeholders
    result_value: str | None = None
    lr_pairs: dict[str, dict[str, float]] = {}  # dx → {lr_pos, lr_neg}
    trajectory: list[dict[str, Any]] = []        # [{t_min, value}, ...]
    source: str = "scripted"
    choosing_wisely: bool = False


class FindingsLibrary:
    """In-memory keyed store of scripted clinical findings."""

    def __init__(self, entries: dict[str, FindingEntry]) -> None:
        self._entries = entries

    @classmethod
    def load(cls, path: Path | None = None) -> "FindingsLibrary":
        path = path or settings.findings_lib_path
        if not path.exists():
            return cls({})
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        entries = {k: FindingEntry.model_validate({"finding_id": k, **v})
                   for k, v in raw.items()}
        return cls(entries)

    def resolve(self, finding_id: str, elapsed_minutes: float = 0) -> FindingEntry | None:
        entry = self._entries.get(finding_id)
        if entry is None:
            return None
        # Apply time trajectory if defined
        if entry.trajectory:
            val = entry.trajectory[0]["value"]
            for point in entry.trajectory:
                if elapsed_minutes >= point["t_min"]:
                    val = point["value"]
            entry = entry.model_copy(update={"result_value": str(val)})
        return entry

    def get_lr(self, finding_id: str, dx: str) -> dict[str, float]:
        entry = self._entries.get(finding_id)
        if entry:
            return entry.lr_pairs.get(dx, {})
        return {}

    def all_ids(self) -> list[str]:
        return list(self._entries.keys())
