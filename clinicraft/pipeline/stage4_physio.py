"""
Stage 4: Physiological Grounding
Framework: Pulse Engine (khronos-medical/pulse) + Findings Library fallback.

Hybrid Grounding decision tree (§6 Difficulty 1):
  finding ∈ {vitals, hemodynamics, resp mechanics, gas exchange, basic PK/PD}
      → live (Pulse Engine, dynamic, responds to treatment)
  otherwise
      → scripted (Findings Library, LR-tagged, time-trajectory-capable)

Pulse Engine Python SDK: pulse-cpp-python (install from Pulse GitHub releases).
Falls back to MockPulseClient if SDK is unavailable.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from clinicraft.schemas.ground_truth import GroundTruthGraph
from clinicraft.physio.grounding import HybridGrounder


def parse_case_vitals(vitals: dict[str, Any]) -> dict[str, Any]:
    """Normalise a ClinicalCase.vitals dict into grounder keys (SBP/DBP split)."""
    out: dict[str, Any] = {}
    for k, v in (vitals or {}).items():
        if k == "BP" and isinstance(v, str) and "/" in v:
            try:
                sbp, dbp = v.split("/")[:2]
                out["SBP"] = float(sbp.strip())
                out["DBP"] = float(dbp.strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif k in ("HR", "RR", "SpO2", "T", "SBP", "DBP", "GCS"):
            try:
                out[k] = float(str(v).split()[0].rstrip("%"))
            except (ValueError, IndexError):
                pass
    return out


async def ground_case(
    gtg: GroundTruthGraph,
    initial_vitals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assign each GTG finding a grounding source (live vs scripted).
    Returns a world_config physio block ready for CasePack.
    `initial_vitals` should already be in grounder-key form (see parse_case_vitals).
    """
    grounder = HybridGrounder()
    result = grounder.ground(gtg, initial_vitals or {})

    logger.info(
        f"[{gtg.case_id}] Physiological grounding: "
        f"live={result['live_count']}, scripted={result['scripted_count']}, "
        f"dynamic_coverage={result['dynamic_coverage']:.2f}"
    )
    return result
