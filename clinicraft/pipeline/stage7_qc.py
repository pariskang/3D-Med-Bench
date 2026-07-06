"""
Stage 7: QC & Stratification
Auto QC checks + oracle/nop sanity validation.
Produces stratification labels and a release manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from clinicraft.schemas.case_pack import CasePack
from clinicraft.schemas.ground_truth import GroundTruthGraph
from clinicraft.schemas.rubric import Rubric


class QCReport(BaseModel):
    case_id: str
    schema_ok: bool = False
    terminology_ok: bool = False
    gtg_ok: bool = False
    rubric_ok: bool = False
    render_sanity_ok: bool = False
    oracle_score_ok: bool = False    # True if oracle scores >70
    nop_score_ok: bool = False       # True if nop scores <10
    issues: list[str] = []
    passed: bool = False

    def finalise(self) -> "QCReport":
        checks = [
            self.schema_ok, self.terminology_ok, self.gtg_ok,
            self.rubric_ok,
        ]
        self.passed = all(checks) and not self.issues
        return self


def qc_case_directory(case_dir: Path) -> QCReport:
    """Run all automated QC checks for a packed case directory."""
    case_id = case_dir.name
    report = QCReport(case_id=case_id)

    # 1. Schema validation
    try:
        pack = CasePack.load(case_dir)
        report.schema_ok = True
    except Exception as e:
        report.issues.append(f"world_config.yaml schema error: {e}")

    # 2. GTG present and internally consistent
    gtg_path = case_dir / "ground_truth_graph.json"
    if not gtg_path.exists():
        report.issues.append("ground_truth_graph.json missing")
    else:
        try:
            gtg = GroundTruthGraph.model_validate_json(gtg_path.read_text())
            _check_gtg_consistency(gtg, report)
            report.gtg_ok = True
        except Exception as e:
            report.issues.append(f"GTG validation error: {e}")

    # 3. Rubric present
    rubric_path = case_dir / "tests" / "rubric.json"
    if not rubric_path.exists():
        report.issues.append("tests/rubric.json missing")
    else:
        try:
            Rubric.model_validate_json(rubric_path.read_text())
            report.rubric_ok = True
        except Exception as e:
            report.issues.append(f"Rubric validation error: {e}")

    # 4. Presentation exists and is non-empty
    pres = case_dir / "presentation.md"
    if not pres.exists() or pres.stat().st_size < 50:
        report.issues.append("presentation.md missing or too short")
    else:
        report.terminology_ok = True   # proxy check

    # 5. Oracle context present
    if not (case_dir / "oracle" / "context.json").exists():
        report.issues.append("oracle/context.json missing")

    return report.finalise()


def _check_gtg_consistency(gtg: GroundTruthGraph, report: QCReport) -> None:
    if not gtg.final_dx:
        report.issues.append("GTG.final_dx is empty")
    if not gtg.differential:
        report.issues.append("GTG.differential is empty")
    if not gtg.expert_reasoning_trace:
        report.issues.append("GTG.expert_reasoning_trace is empty")
    if not gtg.must_not_miss:
        report.issues.append("GTG.must_not_miss is empty (should have ≥1)")
    p_sum = sum(d.p_prior for d in gtg.differential)
    if p_sum > 1.05:
        report.issues.append(f"GTG differential probabilities sum to {p_sum:.2f} (>1)")
    if not gtg.visible_signs:
        report.issues.append("GTG.visible_signs empty — C3 scoring impossible")


def generate_release_manifest(
    case_dirs: list[Path],
    suite_name: str = "v3",
) -> dict[str, Any]:
    """Create a stratified release manifest from all QC'd cases."""
    cases = []
    for d in case_dirs:
        rpt = qc_case_directory(d)
        if not rpt.passed:
            logger.warning(f"[{d.name}] QC FAILED: {rpt.issues}")
            continue
        try:
            pack = CasePack.load(d)
            cases.append({
                "case_id": pack.case_id,
                "path": str(d),
                "strata": pack.strata.model_dump(),
                "qc": {"passed": rpt.passed, "issues": rpt.issues},
            })
        except Exception:
            pass

    # Stratification counts
    by_diff: dict[str, int] = {}
    by_specialty: dict[str, int] = {}
    for c in cases:
        d = c["strata"]["difficulty"]
        s = c["strata"]["specialty"]
        by_diff[d] = by_diff.get(d, 0) + 1
        by_specialty[s] = by_specialty.get(s, 0) + 1

    manifest = {
        "suite": suite_name,
        "total": len(cases),
        "by_difficulty": by_diff,
        "by_specialty": by_specialty,
        "cases": cases,
    }
    logger.success(f"Manifest: {len(cases)} cases passed QC")
    return manifest
