"""
Replay system — deterministic re-run of a saved encounter trace.
Enables audit, interrater reliability, and headless evidence capture.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from clinicraft.schemas.interaction import Action, Observation


class ReplayPlayer:
    """
    Reads a saved encounter trace and replays actions step by step.
    The environment is deterministic (seeded) so replaying the same
    trace produces identical observations.
    """

    def __init__(self, trace_path: Path) -> None:
        self._trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self._turns = self._trace.get("turns", [])
        self._cursor = 0

    @property
    def total_turns(self) -> int:
        return len(self._turns)

    def has_next(self) -> bool:
        return self._cursor < len(self._turns)

    def next_action(self) -> Action | None:
        if not self.has_next():
            return None
        turn = self._turns[self._cursor]
        self._cursor += 1
        return Action.model_validate(turn["action"])

    def get_observation(self, turn: int) -> Observation | None:
        for t in self._turns:
            if t["turn"] == turn:
                return Observation.model_validate(t["observation"])
        return None

    def summary(self) -> dict:
        return {
            "case_id": self._trace.get("case_id"),
            "model_id": self._trace.get("model_id"),
            "total_turns": self.total_turns,
            "tests_ordered": self._trace.get("tests_ordered", 0),
            "final_submission": self._trace.get("final_submission"),
        }
