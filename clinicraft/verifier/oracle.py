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
    Should achieve the maximum achievable score (§8 upper-bound calibration anchor).

    Builds a full scripted plan once, then emits one action per turn. Exercises
    every scored dimension: problem rep (C2), perception of visible signs (C3),
    ideal workup (C4), differential + diagnosis (C1/C2), management + safety-net
    (C4). Driven by an explicit queue — no fragile turn arithmetic.
    """

    def __init__(self, oracle_ctx: dict, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._ctx = oracle_ctx
        self._client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._plan: list[Action] = self._build_plan()
        self._cursor = 0

    @classmethod
    def from_case_dir(cls, case_dir: Path, **kwargs) -> "OracleAgent":
        ctx_path = case_dir / "oracle" / "context.json"
        ctx = json.loads(ctx_path.read_text())
        return cls(oracle_ctx=ctx, **kwargs)

    def _build_plan(self) -> list[Action]:
        from clinicraft.schemas.interaction import DifferentialEntry

        ctx = self._ctx
        plan: list[Action] = []

        # 1. Problem representation (C2)
        plan.append(Action(
            action=ActionType.SUBMIT_PROBLEM_REP,
            params={"text": ctx.get("problem_representation", "")},
        ))

        # 2. Perceive/examine each visible sign (C3) — inspect its region.
        for sign in ctx.get("visible_signs", []):
            plan.append(Action(
                action=ActionType.INSPECT,
                params={"region": sign.get("region") or sign.get("description", "")},
            ))

        # 3. Order the ideal workup (C4)
        for step in ctx.get("ideal_workup", []):
            plan.append(Action.order_test(step["test"], step.get("rationale", "")))

        # 4. Differential (C1/C2)
        diff = ctx.get("differential", [])
        if diff:
            plan.append(Action.submit_differential([
                DifferentialEntry(dx=d["dx"], p=d.get("p_prior", d.get("p", 0.1)))
                for d in diff[:5]
            ]))

        # 5. Management steps (C4)
        for m in ctx.get("management_plan", []):
            plan.append(Action(
                action=ActionType.PRESCRIBE,
                params={"plan": m.get("action", ""), "rationale": m.get("rationale", "")},
            ))

        # 6. Safety-net (C4/completeness)
        if ctx.get("safety_net_items"):
            plan.append(Action(
                action=ActionType.SAFETY_NET,
                params={"instructions": ctx["safety_net_items"]},
            ))

        # 7. Terminal diagnosis (C1) — well-calibrated confidence for C6.
        plan.append(Action.submit_diagnosis(
            dx=ctx.get("final_dx", "诊断未知"),
            confidence=0.85,
            rationale=" → ".join(ctx.get("expert_reasoning_trace", [])[:3]),
        ))
        return plan

    async def act(self, observation: dict) -> Action:
        if self._cursor < len(self._plan):
            action = self._plan[self._cursor]
            self._cursor += 1
            return action
        # Plan exhausted (shouldn't happen before terminal) — submit diagnosis.
        return Action.submit_diagnosis(
            dx=self._ctx.get("final_dx", "诊断未知"),
            confidence=0.85,
            rationale="oracle terminal",
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
