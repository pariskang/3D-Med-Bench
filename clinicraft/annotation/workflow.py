"""
Annotation workflow orchestration.

File-based, offline-verifiable review process:
  1. create_tasks:  scan packaged cases → emit a blank YAML review form per
                    (case, annotator) into an annotations directory.
  2. (experts fill the YAML forms — accept/edit/reject + strata labels)
  3. load_case_annotations: read filled forms for one case.
  4. run_irr:       aggregate κ across all annotated cases.
  5. finalize_case: merge annotations (+ arbitration) → write validated GTG back
                    into the case directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from loguru import logger

from clinicraft.annotation.consensus import ConsensusResult, merge_annotations
from clinicraft.annotation.irr import IRRReport, compute_irr
from clinicraft.annotation.schema import (
    AnnotationTask, FieldJudgment, GTGAnnotation, REVIEWABLE_FIELDS,
    StrataJudgment, Verdict,
)
from clinicraft.schemas.ground_truth import GroundTruthGraph


def _load_gtg(case_dir: Path) -> GroundTruthGraph:
    return GroundTruthGraph.model_validate_json(
        (case_dir / "ground_truth_graph.json").read_text(encoding="utf-8")
    )


def build_task(gtg: GroundTruthGraph) -> AnnotationTask:
    return AnnotationTask(
        case_id=gtg.case_id,
        specialty=gtg.specialty,
        draft_final_dx=gtg.final_dx,
        draft_problem_representation=gtg.problem_representation,
        draft_differential=[d.model_dump() for d in gtg.differential],
        draft_must_not_miss=gtg.must_not_miss,
        draft_red_flags=gtg.red_flags,
        draft_difficulty=gtg.difficulty,
        draft_rarity=gtg.rarity,
        draft_error_prone=gtg.error_prone,
    )


def blank_form(gtg: GroundTruthGraph, annotator_id: str) -> dict:
    """A pre-filled YAML review form for an expert to edit."""
    task = build_task(gtg)
    return {
        "case_id": gtg.case_id,
        "annotator_id": annotator_id,
        "role": "specialist",
        "specialty": gtg.specialty,
        "_draft_for_reference": {
            "final_dx": task.draft_final_dx,
            "problem_representation": task.draft_problem_representation,
            "differential": task.draft_differential,
            "must_not_miss": task.draft_must_not_miss,
            "red_flags": task.draft_red_flags,
        },
        "_instructions": task.instructions,
        # --- expert fills below ---
        "overall_valid": True,
        "final_dx_agree": True,
        "final_dx_corrected": None,
        "strata": {"difficulty": gtg.difficulty, "rarity": gtg.rarity,
                   "error_prone": gtg.error_prone},
        "field_judgments": {
            f: {"verdict": "accept", "corrected_value": None, "comment": ""}
            for f in REVIEWABLE_FIELDS
        },
        "free_comment": "",
    }


def create_tasks(
    cases_dir: Path,
    out_dir: Path,
    annotators: list[str],
) -> list[Path]:
    """Emit a blank review form per (case, annotator). Returns written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for gtg_path in sorted(cases_dir.glob("**/ground_truth_graph.json")):
        gtg = GroundTruthGraph.model_validate_json(gtg_path.read_text(encoding="utf-8"))
        for annotator in annotators:
            form = blank_form(gtg, annotator)
            dest = out_dir / gtg.case_id / f"{annotator}.yaml"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                yaml.dump(form, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            written.append(dest)
    logger.info(f"Wrote {len(written)} annotation forms → {out_dir}")
    return written


def parse_form(path: Path) -> GTGAnnotation:
    """Parse a filled YAML review form into a GTGAnnotation."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fj = {}
    for name, j in (data.get("field_judgments") or {}).items():
        fj[name] = FieldJudgment(
            verdict=Verdict(j.get("verdict", "accept")),
            corrected_value=j.get("corrected_value"),
            comment=j.get("comment", ""),
        )
    strata = data.get("strata") or {}
    return GTGAnnotation(
        case_id=data["case_id"],
        annotator_id=data["annotator_id"],
        role=data.get("role", "specialist"),
        specialty=data.get("specialty", ""),
        overall_valid=bool(data.get("overall_valid", True)),
        final_dx_agree=bool(data.get("final_dx_agree", True)),
        final_dx_corrected=data.get("final_dx_corrected"),
        strata=StrataJudgment(
            difficulty=strata.get("difficulty", "hard"),
            rarity=strata.get("rarity", "uncommon"),
            error_prone=bool(strata.get("error_prone", False)),
        ),
        field_judgments=fj,
        free_comment=data.get("free_comment", ""),
    )


def load_case_annotations(
    annotations_dir: Path, case_id: str
) -> tuple[list[GTGAnnotation], GTGAnnotation | None]:
    """
    Load all expert annotations for one case.
    A form whose role == "arbitrator" is returned separately.
    """
    case_dir = annotations_dir / case_id
    experts: list[GTGAnnotation] = []
    arbitrator: GTGAnnotation | None = None
    for form in sorted(case_dir.glob("*.yaml")):
        ann = parse_form(form)
        if ann.role == "arbitrator":
            arbitrator = ann
        else:
            experts.append(ann)
    return experts, arbitrator


def run_irr(annotations_dir: Path, threshold: float = 0.8) -> IRRReport:
    """Aggregate κ across every case that has ≥2 expert annotations."""
    by_case: dict[str, list[GTGAnnotation]] = {}
    for case_dir in sorted(p for p in annotations_dir.iterdir() if p.is_dir()):
        experts, _ = load_case_annotations(annotations_dir, case_dir.name)
        if len(experts) >= 2:
            by_case[case_dir.name] = experts
    return compute_irr(by_case, threshold=threshold)


def finalize_case(
    case_dir: Path,
    annotations_dir: Path,
    write: bool = True,
) -> ConsensusResult:
    """Merge annotations for a case and (optionally) write the validated GTG."""
    gtg = _load_gtg(case_dir)
    experts, arbitrator = load_case_annotations(annotations_dir, gtg.case_id)
    if len(experts) < 2:
        raise ValueError(f"{gtg.case_id}: need ≥2 expert annotations, found {len(experts)}")

    result = merge_annotations(gtg, experts, arbitrator)

    if write:
        out = case_dir / "ground_truth_graph.json"
        out.write_text(
            json.dumps(result.validated_gtg.model_dump(mode="json"),
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Audit trail
        audit = case_dir / "oracle" / "validation_audit.json"
        audit.parent.mkdir(exist_ok=True)
        audit.write_text(json.dumps({
            "validated": result.validated,
            "experts": result.experts,
            "arbitrator": result.arbitrator,
            "unresolved": result.unresolved,
            "disagreements": [
                {"field": d.field_name, "verdicts": d.verdicts,
                 "resolved_by": d.resolved_by}
                for d in result.disagreements
            ],
            "apply_errors": result.apply_errors,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
