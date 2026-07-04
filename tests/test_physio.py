"""
Tests for the literature-grounded physiological scenario dataset + client.
Verifies trajectory values, treatment-response direction, and determinism.
"""

import asyncio
import math
from pathlib import Path

import pytest

from clinicraft.physio.scenario import ScenarioLibrary, PhysioScenario
from clinicraft.physio.dataset_client import DatasetPulseClient

SCEN_DIR = Path("resources/physio_scenarios")


def _lib():
    return ScenarioLibrary.load(SCEN_DIR)


# --------------------------------------------------------------------------
# Dataset integrity
# --------------------------------------------------------------------------

def test_all_scenarios_load_and_validate():
    lib = _lib()
    ids = lib.all_ids()
    assert {"hemorrhage_class3", "septic_shock", "anaphylaxis",
            "respiratory_failure_type2", "cardiogenic_shock_stemi", "dka"} <= set(ids)
    for sid in ids:
        sc = lib.get(sid)
        assert isinstance(sc, PhysioScenario)
        assert sc.source                       # every scenario cites a source
        assert sc.untreated_trajectory
        assert sc.treatments


def test_scenario_selection_by_condition():
    lib = _lib()
    assert lib.select_for_condition("失血性休克", "创伤性失血").scenario_id == "hemorrhage_class3"
    assert lib.select_for_condition("septic shock", "sepsis").scenario_id == "septic_shock"
    assert lib.select_for_condition("急性心肌梗死 心源性休克", "STEMI").scenario_id == "cardiogenic_shock_stemi"
    assert lib.select_for_condition("胸痛待查", "") is None


# --------------------------------------------------------------------------
# Untreated trajectory
# --------------------------------------------------------------------------

def test_untreated_interpolation_matches_reference():
    lib = _lib()
    client = DatasetPulseClient(lib.get("hemorrhage_class3"))

    async def _go():
        await client.initialise("hemorrhage_class3", {})
        s0 = await client.get_state()          # t=0
        # advance to t=15 min (between t=10 SBP 78 and t=20 SBP 68 → 73)
        await client.advance(15 * 60)
        s15 = await client.get_state()
        return s0, s15
    s0, s15 = asyncio.run(_go())
    assert s0["HR"] == pytest.approx(125, abs=0.5)
    assert s0["SBP"] == pytest.approx(88, abs=0.5)
    assert s15["SBP"] == pytest.approx(73, abs=0.5)   # linear midpoint


def test_untreated_deteriorates():
    """Hemorrhage without control → SBP falls, HR rises."""
    client = DatasetPulseClient(_lib().get("hemorrhage_class3"))

    async def _go():
        await client.initialise("hemorrhage_class3", {})
        await client.advance(30 * 60)
        return await client.get_state()
    s = asyncio.run(_go())
    assert s["SBP"] < 60          # trajectory endpoint 55
    assert s["HR"] > 150          # trajectory endpoint 160


# --------------------------------------------------------------------------
# Treatment response
# --------------------------------------------------------------------------

def test_treatment_reverses_shock():
    """Blood transfusion + haemorrhage control → SBP recovers toward target."""
    client = DatasetPulseClient(_lib().get("hemorrhage_class3"))

    async def _go():
        await client.initialise("hemorrhage_class3", {})
        await client.advance(2 * 60)                       # t=2
        sbp_before = (await client.get_state())["SBP"]
        await client.apply_action("prescribe", {"plan": "输血 blood transfusion"})
        await client.apply_action("order_procedure", {"action": "紧急手术止血"})
        await client.advance(20 * 60)                      # t=22
        sbp_after = (await client.get_state())["SBP"]
        return sbp_before, sbp_after
    before, after = asyncio.run(_go())
    assert after > before + 20        # decisively improved
    assert after > 100                # approaching control target (~116)


def test_partial_treatment_unaddressed_vital_still_worsens():
    """
    Oxygen in a haemorrhage case addresses nothing in the bleed → SBP keeps
    falling (honest partial-treatment model).
    """
    client = DatasetPulseClient(_lib().get("hemorrhage_class3"))

    async def _go():
        await client.initialise("hemorrhage_class3", {})
        await client.apply_action("prescribe", {"drug": "oxygen 吸氧"})  # no matching treatment
        await client.advance(20 * 60)
        return await client.get_state()
    s = asyncio.run(_go())
    assert s["SBP"] < 70   # unhelped, continues toward the untreated endpoint


def test_epinephrine_reverses_anaphylaxis_fast():
    client = DatasetPulseClient(_lib().get("anaphylaxis"))

    async def _go():
        await client.initialise("anaphylaxis", {})
        await client.apply_action("prescribe", {"drug": "肾上腺素 epinephrine"})
        await client.advance(8 * 60)
        return await client.get_state()
    s = asyncio.run(_go())
    assert s["SBP"] > 100      # rapid recovery toward 116
    assert s["SpO2"] > 93


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_deterministic_replay():
    async def _run():
        c = DatasetPulseClient(_lib().get("septic_shock"))
        await c.initialise("septic_shock", {})
        await c.advance(10 * 60)
        await c.apply_action("prescribe", {"drug": "去甲肾上腺素"})
        await c.advance(10 * 60)
        return await c.get_state()
    s1 = asyncio.run(_run())
    s2 = asyncio.run(_run())
    assert s1 == s2


def test_factory_returns_dataset_client():
    from clinicraft.physio.pulse_client import get_pulse_client
    client = get_pulse_client("septic_shock")
    assert isinstance(client, DatasetPulseClient)
    assert client.scenario_id == "septic_shock"
