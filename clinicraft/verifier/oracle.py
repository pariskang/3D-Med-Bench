"""
Oracle and Nop agents — §8 upper/lower bounds for score calibration.

oracle: given full GTG context → should score >70 (upper bound)
nop:    does nothing / answers randomly → should score ≈ 0 (lower bound)
human:  human doctor baseline (interactive, not automated)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

from clinicraft.config import settings
from clinicraft.schemas.interaction import Action, ActionType


class OracleAgent:
    """
    Oracle doctor: given access to ground_truth_graph.json and oracle/context.json.
    Should achieve the maximum achievable score.
    """

    def __init__(self, oracle_ctx: dict, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._ctx = oracle_ctx
        self._client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._step = 0
        self._workup_done: set[str] = set()

    @classmethod
    def from_case_dir(cls, case_dir: Path, **kwargs) -> "OracleAgent":
        ctx_path = case_dir / "oracle" / "context.json"
        ctx = json.loads(ctx_path.read_text())
        return cls(oracle_ctx=ctx, **kwargs)

    async def act(self, observation: dict) -> Action:
        """Follow the ideal workup from GTG, then submit correct diagnosis."""
        self._step += 1
        turn = observation.get("turn", self._step)

        # Step 1: Submit problem representation
        if turn == 1:
            return Action(
                action=ActionType.SUBMIT_PROBLEM_REP,
                params={"text": self._ctx.get("problem_representation", "")},
            )

        # Step 2-N: Work through ideal workup
        workup = self._ctx.get("ideal_workup", [])
        remaining = [w for w in workup if w["test"] not in self._workup_done]
        if remaining:
            next_test = remaining[0]
            self._workup_done.add(next_test["test"])
            test_name = next_test["test"]
            if any(kw in test_name for kw in ["CT", "MRI", "X线", "超声", "Echo"]):
                return Action.order_test(test_name, next_test.get("rationale", ""))
            return Action.order_test(test_name, next_test.get("rationale", ""))

        # After workup: submit correct differential
        diff = self._ctx.get("differential", [])
        if diff and turn == len(workup) + 2:
            from clinicraft.schemas.interaction import DifferentialEntry
            return Action.submit_differential([
                DifferentialEntry(dx=d["dx"], p=d["p_prior"])
                for d in diff[:5]
            ])

        # Final: submit diagnosis
        return Action.submit_diagnosis(
            dx=self._ctx.get("final_dx", "诊断未知"),
            confidence=0.9,
            rationale=" → ".join(self._ctx.get("expert_reasoning_trace", [])[:3]),
        )


class NopAgent:
    """
    Nop (no-operation) agent: does nothing useful.
    Expected score: ≈0. Used as lower bound in calibration.
    """

    async def act(self, observation: dict) -> Action:
        turn = observation.get("turn", 1)
        if turn == 1:
            return Action(action=ActionType.ASK, params={"utterance": "你好"})
        if turn >= 5:
            return Action.submit_diagnosis(
                dx="感冒", confidence=0.1, rationale="随机猜测"
            )
        return Action(action=ActionType.EXPRESS_UNCERTAINTY,
                      params={"confidence": 0.1, "would_defer": True})
