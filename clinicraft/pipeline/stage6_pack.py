"""
Stage 6: Packaging
Assembles a fully playable case directory from all prior stage outputs.

Output structure (per Appendix D):
  cases/<specialty>/<case_id>/
    presentation.md        ← only this is visible to the SUT at runtime
    world_config.yaml
    ground_truth_graph.json
    tests/rubric.json
    resources/             ← curated imaging/labs (refs)
    oracle/                ← oracle agent context (hidden from SUT)
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from loguru import logger

from clinicraft.config import settings
from clinicraft.pipeline.stage3_gtg import auto_generate_rubric_requirements
from clinicraft.schemas.case_pack import CasePack, PhysioConfig, Strata, Source, WorldConfig
from clinicraft.schemas.clinical_case import ClinicalCase
from clinicraft.schemas.ground_truth import GroundTruthGraph
from clinicraft.schemas.rubric import (
    CompletenessCheck, Penalties, Rubric, RubricRequirement,
    ScoreFormula, DimensionWeights,
)


def pack_case(
    case: ClinicalCase,
    gtg: GroundTruthGraph,
    physio_result: dict,
    embody_result: dict,
    out_root: Path | None = None,
    seed: int = 42,
) -> Path:
    """
    Write all case files to disk. Returns the case directory path.
    """
    out_root = out_root or settings.cases_dir
    specialty_slug = (gtg.specialty or "general").lower().replace(" ", "_")
    case_dir = out_root / specialty_slug / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "tests").mkdir(exist_ok=True)
    (case_dir / "resources").mkdir(exist_ok=True)
    (case_dir / "oracle").mkdir(exist_ok=True)

    # --- presentation.md (only file visible to SUT) ---
    _write_presentation(case_dir, case, gtg)

    # --- ground_truth_graph.json (hidden from SUT) ---
    gtg_path = case_dir / "ground_truth_graph.json"
    gtg_path.write_text(
        json.dumps(gtg.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- world_config.yaml ---
    pack = CasePack(
        case_id=case.case_id,
        source=Source(
            provenance="private_hospital",
            deid_standard="GB/T42460-2023",
            contamination_free=True,
        ),
        strata=Strata(
            difficulty=gtg.difficulty,
            rarity=gtg.rarity,
            error_prone=gtg.error_prone,
            specialty=gtg.specialty,
            perception_tier=embody_result.get("perception_tier", "T2"),
            dynamic_coverage=physio_result.get("dynamic_coverage", 0.0),
        ),
        world_config=WorldConfig(
            seed=seed,
            physio=PhysioConfig(
                engine=physio_result.get("engine", "scripted"),
                dynamic_coverage=physio_result.get("dynamic_coverage", 0.0),
                initial_state=physio_result.get("initial_state", {}),
            ),
            available_tests=_collect_available_tests(gtg),
            available_imaging=_collect_available_imaging(gtg),
        ),
    )
    pack.save(case_dir)

    # --- tests/rubric.json ---
    rubric = _build_rubric(case.case_id, gtg)
    rubric_path = case_dir / "tests" / "rubric.json"
    rubric_path.write_text(
        json.dumps(rubric.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- oracle/context.json (oracle agent gets GTG + ideal workup) ---
    oracle_ctx = {
        "case_id": case.case_id,
        "final_dx": gtg.final_dx,
        "differential": [d.model_dump() for d in gtg.differential],
        "expert_reasoning_trace": gtg.expert_reasoning_trace,
        "ideal_workup": [w.model_dump() for w in gtg.ideal_workup],
        "management_plan": [m.model_dump() for m in gtg.management_plan],
    }
    (case_dir / "oracle" / "context.json").write_text(
        json.dumps(oracle_ctx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.success(f"[{case.case_id}] Packed → {case_dir}")
    return case_dir


def _write_presentation(case_dir: Path, case: ClinicalCase, gtg: GroundTruthGraph) -> None:
    """
    Write presentation.md — the ONLY information the SUT sees at episode start.
    Contains chief complaint + initial vitals. No diagnosis, no workup results.
    """
    lines = [
        f"# 病例 {case.case_id}",
        "",
        "## 主诉",
        case.chief_complaint or "（见患者陈述）",
        "",
        "## 就诊情境",
    ]
    if case.age:
        lines.append(f"患者，年龄段 {case.social_history or '（见档案）'}，")
    if case.vitals:
        lines.append("")
        lines.append("## 初始生命体征")
        for k, v in case.vitals.items():
            lines.append(f"- {k}: {v}")
    lines += [
        "",
        "---",
        "> *请开始问诊。你可以提问、查体、开单检查、或请求上级医师。*",
    ]
    (case_dir / "presentation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _build_rubric(case_id: str, gtg: GroundTruthGraph) -> Rubric:
    raw_reqs = auto_generate_rubric_requirements(gtg)
    reqs = [RubricRequirement.model_validate(r) for r in raw_reqs]
    hard_veto = [r.id for r in reqs if r.veto_if_fail]
    return Rubric(
        case_id=case_id,
        completeness_check=CompletenessCheck(),
        requirements=reqs,
        score_formula=ScoreFormula(
            weights=DimensionWeights(),
            penalties=Penalties(),
            hard_veto_ids=hard_veto,
        ),
    )


def _collect_available_tests(gtg: GroundTruthGraph) -> list[str]:
    return [w.test for w in gtg.ideal_workup if w.loinc is not None or "检" in w.test or "化" in w.test]


def _collect_available_imaging(gtg: GroundTruthGraph) -> list[str]:
    imaging_keywords = {"CT", "MRI", "X线", "胸片", "超声", "Echo", "PET", "造影", "内镜"}
    return [
        w.test for w in gtg.ideal_workup
        if any(kw.lower() in w.test.lower() for kw in imaging_keywords)
    ]
