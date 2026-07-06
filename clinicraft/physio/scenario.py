"""
Physiological scenario schema + loader.

Adopts the Pulse Physiology Engine's data-format contract (JSON scenario files,
Pulse-canonical vital column names, patient-state concept) so the real Kitware
engine drops into the same seam when installed. Until then, a data-driven client
(dataset_client.py) replays these literature-grounded trajectories.

Vital keys follow Pulse conventions where practical:
  HR   → HeartRate (1/min)
  SBP  → SystolicArterialPressure (mmHg)
  DBP  → DiastolicArterialPressure (mmHg)
  RR   → RespirationRate (1/min)
  SpO2 → OxygenSaturation (%)
  T    → CoreTemperature (degC)
  PaCO2, lactate, GCS → extended clinical channels

Trajectories are grounded in published clinical reference data (ATLS hemorrhage
classes, Sepsis-3, anaphylaxis guidelines); each scenario cites its source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from clinicraft.config import settings

VITAL_KEYS = ["HR", "SBP", "DBP", "RR", "SpO2", "T", "PaCO2", "lactate", "GCS"]


class VitalSample(BaseModel):
    """One point on a vital-sign trajectory."""
    t_min: float
    HR: float | None = None
    SBP: float | None = None
    DBP: float | None = None
    RR: float | None = None
    SpO2: float | None = None
    T: float | None = None
    PaCO2: float | None = None
    lactate: float | None = None
    GCS: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {k: v for k in VITAL_KEYS if (v := getattr(self, k)) is not None}


class TreatmentResponse(BaseModel):
    """
    How a therapeutic action bends the trajectory.

    On application, each addressed vital approaches its `target` with a
    first-order time constant `tau_min` (v(t) = target + (v0-target)·e^-(t-t0)/tau).
    Deterministic and physiologically shaped (exponential approach to steady state).
    """
    treatment: str                        # canonical key
    aliases: list[str] = []               # drug/keyword matches from action params
    onset_min: float = 1.0
    tau_min: float = 5.0
    targets: dict[str, float] = {}        # vital → steady-state value it drives toward
    note: str = ""


class PhysioScenario(BaseModel):
    scenario_id: str
    condition: str
    description: str
    source: str                           # literature citation
    baseline: dict[str, float] = {}
    untreated_trajectory: list[VitalSample] = []
    treatments: list[TreatmentResponse] = []
    pulse_scenario_ref: str | None = None  # real Pulse JSON scenario, if present

    def match_treatment(self, action_type: str, params: dict[str, Any]) -> TreatmentResponse | None:
        """Map an environment action to a scenario treatment via aliases."""
        hay = " ".join([
            action_type or "",
            str(params.get("drug", "")), str(params.get("plan", "")),
            str(params.get("action", "")), str(params.get("test", "")),
        ]).lower()
        for tr in self.treatments:
            if any(alias.lower() in hay for alias in tr.aliases):
                return tr
        return None


class ScenarioLibrary:
    """Loads all physiological scenarios from resources/physio_scenarios/."""

    def __init__(self, scenarios: dict[str, PhysioScenario]) -> None:
        self._scenarios = scenarios

    @classmethod
    def load(cls, path: Path | None = None) -> "ScenarioLibrary":
        path = path or (settings.resources_dir / "physio_scenarios")
        scenarios: dict[str, PhysioScenario] = {}
        if path.exists():
            for f in sorted(path.glob("*.yaml")):
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                sc = PhysioScenario.model_validate(data)
                scenarios[sc.scenario_id] = sc
        return cls(scenarios)

    def get(self, scenario_id: str) -> PhysioScenario | None:
        return self._scenarios.get(scenario_id)

    def all_ids(self) -> list[str]:
        return list(self._scenarios.keys())

    def select_for_condition(self, condition: str, final_dx: str = "") -> PhysioScenario | None:
        """Pick the best-matching scenario for a case's condition / diagnosis."""
        text = f"{condition} {final_dx}".lower()
        best: tuple[int, PhysioScenario] | None = None
        for sc in self._scenarios.values():
            score = 0
            for kw in sc.condition.lower().split("|"):
                kw = kw.strip()
                if kw and kw in text:
                    score += len(kw)
            if score and (best is None or score > best[0]):
                best = (score, sc)
        return best[1] if best else None
