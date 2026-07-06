"""
DatasetPulseClient — replays a literature-grounded PhysioScenario deterministically.

Implements the same PulseClientProtocol as the real Kitware engine adapter, so
it is a drop-in physiology backend. Vital trajectories come from the scenario
dataset; therapeutic actions bend addressed vitals toward treatment targets via
a first-order (exponential) approach, while un-addressed vitals keep following
the untreated disease trajectory (partial treatment is modelled honestly).

Fully deterministic: no randomness, no wall-clock — identical (scenario, action
sequence, timestamps) → identical vitals. Safe for seeded replay.
"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger

from clinicraft.physio.scenario import PhysioScenario, ScenarioLibrary, VITAL_KEYS


class _AppliedTreatment:
    __slots__ = ("treatment", "t_applied_min", "snapshot")

    def __init__(self, treatment, t_applied_min: float, snapshot: dict[str, float]) -> None:
        self.treatment = treatment
        self.t_applied_min = t_applied_min
        self.snapshot = snapshot


class DatasetPulseClient:
    """Data-driven physiology backend replaying a PhysioScenario."""

    def __init__(self, scenario: PhysioScenario | None = None) -> None:
        self._scenario = scenario
        self._t_min = 0.0
        self._applied: list[_AppliedTreatment] = []
        self._baseline: dict[str, float] = {}
        self._is_photoreal = False

    @property
    def scenario_id(self) -> str | None:
        return self._scenario.scenario_id if self._scenario else None

    async def initialise(self, scenario_ref: str, initial_state: dict) -> bool:
        """
        scenario_ref: a scenario_id resolvable in the ScenarioLibrary (or "").
        initial_state: constant fallback vitals (e.g. from the case presentation).
        """
        if self._scenario is None and scenario_ref:
            self._scenario = ScenarioLibrary.load().get(scenario_ref)
        if self._scenario:
            self._baseline = dict(self._scenario.baseline)
        self._baseline.update({k: float(v) for k, v in (initial_state or {}).items()
                               if isinstance(v, (int, float))})
        self._t_min = 0.0
        self._applied = []
        if self._scenario:
            logger.info(f"DatasetPulseClient: scenario '{self._scenario.scenario_id}' "
                        f"({self._scenario.condition})")
        else:
            logger.info("DatasetPulseClient: no scenario (constant baseline mode)")
        return True

    async def advance(self, dt_seconds: float = 60.0) -> dict[str, Any]:
        self._t_min += dt_seconds / 60.0
        return await self.get_state()

    async def apply_action(self, action_type: str, params: dict) -> bool:
        if not self._scenario:
            return False
        tr = self._scenario.match_treatment(action_type, params)
        if tr is None:
            return False
        snapshot = self._compute_state(self._t_min)
        self._applied.append(_AppliedTreatment(tr, self._t_min, snapshot))
        logger.debug(f"treatment '{tr.treatment}' applied at t={self._t_min:.1f}min")
        return True

    async def get_state(self) -> dict[str, Any]:
        state = self._compute_state(self._t_min)
        state["sim_time_s"] = self._t_min * 60.0
        return state

    async def shutdown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Trajectory computation
    # ------------------------------------------------------------------

    def _compute_state(self, t_min: float) -> dict[str, float]:
        state: dict[str, float] = {}
        keys = set(self._baseline)
        if self._scenario:
            for s in self._scenario.untreated_trajectory:
                keys.update(s.as_dict().keys())
        for vital in keys:
            state[vital] = round(self._current(vital, t_min), 2)
        return state

    def _current(self, vital: str, t_min: float) -> float:
        # Most recent applied treatment that addresses this vital.
        active = None
        for ap in self._applied:
            if vital in ap.treatment.targets and t_min >= ap.t_applied_min:
                if active is None or ap.t_applied_min >= active.t_applied_min:
                    active = ap
        if active is not None:
            tr = active.treatment
            v0 = active.snapshot.get(vital, self._untreated(vital, active.t_applied_min))
            target = tr.targets[vital]
            elapsed = t_min - active.t_applied_min - tr.onset_min
            if elapsed <= 0:
                return v0
            return target + (v0 - target) * math.exp(-elapsed / max(tr.tau_min, 1e-6))
        return self._untreated(vital, t_min)

    def _untreated(self, vital: str, t_min: float) -> float:
        """Linear-interpolate the untreated trajectory; fall back to baseline."""
        pts = []
        if self._scenario:
            for s in self._scenario.untreated_trajectory:
                v = getattr(s, vital)
                if v is not None:
                    pts.append((s.t_min, float(v)))
        if not pts:
            return float(self._baseline.get(vital, 0.0))
        if t_min <= pts[0][0]:
            return pts[0][1]
        if t_min >= pts[-1][0]:
            return pts[-1][1]
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= t_min <= t1:
                frac = (t_min - t0) / (t1 - t0) if t1 > t0 else 0.0
                return v0 + (v1 - v0) * frac
        return pts[-1][1]
