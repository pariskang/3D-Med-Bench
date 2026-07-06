"""
Pulse Engine client adapter.
Real client: wraps pulse-cpp-python SDK (khronos-medical/pulse on GitHub).
Mock client: used when SDK is unavailable — returns scripted vitals.

Installation:
  pip install pulse-cpp-python  (or build from source)
  See: https://gitlab.kitware.com/physiology/engine
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from loguru import logger


class PulseClientProtocol(Protocol):
    async def initialise(self, scenario_xml: str, initial_state: dict) -> bool: ...
    async def advance(self, dt_seconds: float) -> dict[str, Any]: ...
    async def apply_action(self, action_type: str, params: dict) -> bool: ...
    async def get_state(self) -> dict[str, Any]: ...
    async def shutdown(self) -> None: ...


class MockPulseClient:
    """
    Scripted physiological mock — used when Pulse SDK is not installed.
    Implements realistic vital-sign trajectories for common scenarios.
    """

    _DEFAULTS: dict[str, Any] = {
        "HR": 88, "SBP": 118, "DBP": 76, "RR": 18,
        "SpO2": 97.0, "T": 37.2, "GCS": 15,
        "CO": 5.2,   # cardiac output L/min
        "SVR": 1200, # systemic vascular resistance
    }

    def __init__(self) -> None:
        self._state: dict[str, Any] = dict(self._DEFAULTS)
        self._sim_time: float = 0.0
        self._actions: list[dict] = []
        logger.warning("Using MockPulseClient — Pulse Engine SDK not installed")

    async def initialise(self, scenario_xml: str, initial_state: dict) -> bool:
        self._state.update(initial_state)
        return True

    async def advance(self, dt_seconds: float = 60.0) -> dict[str, Any]:
        self._sim_time += dt_seconds
        self._apply_decay()
        return dict(self._state) | {"sim_time_s": self._sim_time}

    async def apply_action(self, action_type: str, params: dict) -> bool:
        self._actions.append({"type": action_type, "params": params, "t": self._sim_time})
        self._apply_treatment(action_type, params)
        return True

    async def get_state(self) -> dict[str, Any]:
        return dict(self._state) | {"sim_time_s": self._sim_time}

    async def shutdown(self) -> None:
        pass

    def _apply_decay(self) -> None:
        """Simulate slow deterioration if no treatment — keeps mock realistic."""
        for k, delta in [("SBP", -0.1), ("SpO2", -0.05), ("HR", 0.3)]:
            self._state[k] = max(0, self._state.get(k, 0) + delta)

    def _apply_treatment(self, action_type: str, params: dict) -> None:
        """Simplified treatment response."""
        if action_type == "oxygen_supplementation":
            fio2 = params.get("fio2", 0.28)
            self._state["SpO2"] = min(100.0, self._state["SpO2"] + (fio2 - 0.21) * 80)
        elif action_type == "iv_fluid_bolus":
            vol_ml = params.get("volume_ml", 500)
            self._state["SBP"] = min(160, self._state["SBP"] + vol_ml * 0.04)
        elif action_type == "vasopressor":
            dose = params.get("dose_mcg_kg_min", 5)
            self._state["SBP"] = min(180, self._state["SBP"] + dose * 2)
            self._state["HR"] = min(130, self._state["HR"] + dose)


def get_pulse_client(scenario_id: str | None = None) -> PulseClientProtocol:
    """
    Factory. Preference order:
      1. Real Kitware Pulse Physiology Engine, if its Python bindings are present.
      2. DatasetPulseClient — replays a literature-grounded PhysioScenario
         (data-driven, treatment-responsive, deterministic). Default backend.
      3. MockPulseClient — last-resort constant/decay stub.
    NB: the PyPI names `pulse-engine` and `PyPulse` are UNRELATED packages
    (a social-media framework and a pulsar-astronomy tool); we deliberately do
    not import those. Only the genuine Kitware `pulse.engine.PulseEngine` counts.
    """
    try:
        from pulse.engine.PulseEngine import PulseEngine  # type: ignore[import]
        return _RealPulseClient(PulseEngine())
    except Exception:
        pass
    try:
        from clinicraft.physio.dataset_client import DatasetPulseClient
        from clinicraft.physio.scenario import ScenarioLibrary
        scenario = ScenarioLibrary.load().get(scenario_id) if scenario_id else None
        return DatasetPulseClient(scenario)  # type: ignore[return-value]
    except Exception as e:
        logger.warning(f"DatasetPulseClient unavailable ({e}); using MockPulseClient")
        return MockPulseClient()  # type: ignore[return-value]


class _RealPulseClient:
    """Thin adapter around the real Pulse Python SDK."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    async def initialise(self, scenario_xml: str, initial_state: dict) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._engine.initialize_engine, scenario_xml)

    async def advance(self, dt_seconds: float = 60.0) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._engine.advance_model_time, dt_seconds)
        return await self.get_state()

    async def apply_action(self, action_type: str, params: dict) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._engine.process_action, action_type, params
        )

    async def get_state(self) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._engine.get_state_dict)

    async def shutdown(self) -> None:
        self._engine.shutdown()
