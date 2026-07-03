"""
Consequence engine — maps treatment actions to physiological state changes.
Used by ClinicalEnvironment.step() to close the treatment→physio loop (§5.4).
"""

from __future__ import annotations

from typing import Any

from clinicraft.schemas.interaction import ActionType


_TREATMENT_EFFECTS: dict[str, dict[str, float]] = {
    # action_type → {vital: delta}
    "oxygen_therapy": {"SpO2": +5.0, "RR": -2.0},
    "iv_fluid":       {"SBP": +10.0, "HR": -5.0},
    "vasopressor":    {"SBP": +15.0, "HR": +8.0},
    "nitrate":        {"SBP": -8.0, "HR": +5.0},
    "beta_blocker":   {"HR": -15.0, "SBP": -8.0},
    "diuretic":       {"SBP": -5.0, "SpO2": +3.0},
    "antiepileptic":  {},
    "antibiotics":    {"T": -0.3},
    "analgesic":      {"HR": -5.0, "RR": -2.0},
}

_UNTREATED_DECAY: dict[str, float] = {
    "STEMI":             {"HR": +3.0, "SBP": -5.0, "SpO2": -1.0},
    "sepsis":            {"HR": +4.0, "SBP": -6.0, "T": +0.1},
    "PE":                {"HR": +3.0, "SpO2": -2.0, "RR": +2.0},
    "type2_resp_fail":   {"SpO2": -2.0, "RR": +3.0, "PaCO2": +2.0},
}


def apply_consequence(
    current_vitals: dict[str, Any],
    action_type: str,
    action_params: dict[str, Any],
    primary_dx: str | None = None,
    elapsed_min: int = 0,
) -> dict[str, Any]:
    """
    Compute next-state vitals given an action.
    Also applies untreated-disease decay if action is non-therapeutic.
    """
    vitals = dict(current_vitals)

    # Apply treatment effect
    effect_key = _classify_action(action_type, action_params)
    if effect_key in _TREATMENT_EFFECTS:
        for vital, delta in _TREATMENT_EFFECTS[effect_key].items():
            vitals[vital] = vitals.get(vital, 0) + delta

    # Apply disease decay (untreated deterioration)
    if primary_dx and elapsed_min > 0:
        decay = _UNTREATED_DECAY.get(primary_dx, {})
        decay_scale = elapsed_min / 60.0  # per-hour rate
        for vital, rate in decay.items():
            vitals[vital] = vitals.get(vital, 0) + rate * decay_scale

    # Physiological clamps
    vitals["SpO2"] = min(100.0, max(0.0, vitals.get("SpO2", 95)))
    vitals["HR"] = max(20, min(250, vitals.get("HR", 80)))
    vitals["SBP"] = max(40, min(250, vitals.get("SBP", 120)))

    return vitals


def _classify_action(action_type: str, params: dict) -> str:
    """Heuristically map action + params to a treatment category."""
    drug = params.get("drug", "").lower()
    if "oxygen" in drug or action_type == "oxygen":
        return "oxygen_therapy"
    if any(x in drug for x in ["fluid", "生理盐水", "乳酸", "碳酸氢"]):
        return "iv_fluid"
    if any(x in drug for x in ["去甲肾", "多巴胺", "norepinephrine"]):
        return "vasopressor"
    if any(x in drug for x in ["硝酸", "nitrate"]):
        return "nitrate"
    if any(x in drug for x in ["美托洛尔", "阿替洛尔", "metoprolol"]):
        return "beta_blocker"
    if any(x in drug for x in ["呋塞米", "托拉塞米", "furosemide"]):
        return "diuretic"
    if any(x in drug for x in ["青霉素", "头孢", "万古", "美罗培南"]):
        return "antibiotics"
    return ""
