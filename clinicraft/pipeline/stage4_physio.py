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


async def ground_case(
    gtg: GroundTruthGraph,
    initial_vitals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assign each GTG finding a grounding source (live vs scripted).
    Returns a world_config physio block ready for CasePack.
    """
    grounder = HybridGrounder()
    result = grounder.ground(gtg, initial_vitals or {})

    logger.info(
        f"[{gtg.case_id}] Physiological grounding: "
        f"live={result['live_count']}, scripted={result['scripted_count']}, "
        f"dynamic_coverage={result['dynamic_coverage']:.2f}"
    )
    return result
