"""Hybrid Grounding: assigns each GTG finding to live (Pulse) or scripted."""

from __future__ import annotations

from typing import Any

from clinicraft.schemas.ground_truth import GroundTruthGraph


_PULSE_SIMULATABLE = {
    # vital signs
    "心率", "血压", "呼吸", "血氧", "体温", "脉搏",
    "HR", "BP", "RR", "SpO2", "SBP", "DBP", "T",
    # hemodynamics
    "cardiac output", "CO", "心输出量", "SVR", "外周血管阻力",
    "CVP", "中心静脉压", "PAP", "肺动脉压",
    # respiratory
    "tidal volume", "潮气量", "FiO2", "PEEP", "compliance",
    "分钟通气量", "呼气末", "气道阻力",
    # gas exchange
    "PaO2", "PaCO2", "PaO2/FiO2", "P/F", "氧合", "二氧化碳",
    # basic PK/PD
    "药代", "血药浓度", "半衰期",
}


class HybridGrounder:
    """Assigns GTG workup steps to live or scripted grounding."""

    def ground(
        self,
        gtg: GroundTruthGraph,
        initial_vitals: dict[str, Any],
    ) -> dict[str, Any]:
        live_tests, scripted_tests = [], []

        for step in gtg.ideal_workup:
            if self._is_pulse_simulatable(step.test):
                live_tests.append(step.test)
            else:
                scripted_tests.append(step.test)

        total = len(gtg.ideal_workup) or 1
        dynamic_coverage = round(len(live_tests) / total, 3)

        # Select a literature-grounded physiological scenario for this case.
        from clinicraft.physio.scenario import ScenarioLibrary
        condition_text = " ".join([
            gtg.problem_representation, gtg.final_dx, gtg.specialty,
            " ".join(d.dx for d in gtg.differential),
        ])
        scenario = ScenarioLibrary.load().select_for_condition(condition_text, gtg.final_dx)
        scenario_id = scenario.scenario_id if scenario else None

        # Engine: dataset when a scenario matched, else scripted.
        engine = "dataset" if scenario_id else ("pulse" if live_tests else "scripted")

        # Build initial state from GTG initial vitals + case vitals
        initial_state: dict[str, Any] = {
            "HR": initial_vitals.get("HR", 88),
            "SBP": initial_vitals.get("SBP", 118),
            "DBP": initial_vitals.get("DBP", 76),
            "SpO2": initial_vitals.get("SpO2", 97),
            "RR": initial_vitals.get("RR", 18),
            "T": initial_vitals.get("T", 37.2),
        }

        return {
            "engine": engine,
            "scenario_id": scenario_id,
            "live_tests": live_tests,
            "scripted_tests": scripted_tests,
            "live_count": len(live_tests),
            "scripted_count": len(scripted_tests),
            "dynamic_coverage": dynamic_coverage,
            "initial_state": initial_state,
        }

    def _is_pulse_simulatable(self, test_name: str) -> bool:
        test_lower = test_name.lower()
        return any(kw.lower() in test_lower for kw in _PULSE_SIMULATABLE)
