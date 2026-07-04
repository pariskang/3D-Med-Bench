"""
Consensus & arbitration — merge expert annotations into a validated GTG.

Rules per field:
  - all annotators ACCEPT           → keep the draft value
  - unanimous EDIT to the same value → apply the correction
  - anything else (REJECT / conflicting EDITs) → DISAGREEMENT
      → resolved by the arbitrator's judgment if one is supplied, else left
        unresolved and the case is NOT marked validated.

Stratification labels (difficulty/rarity/error_prone) use majority vote; ties
are broken by the arbitrator, else flagged.

A case is validated=True only when there are no unresolved disagreements and the
panel's consensus on `overall_valid` is True.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from clinicraft.annotation.schema import GTGAnnotation, REVIEWABLE_FIELDS, Verdict
from clinicraft.schemas.ground_truth import GroundTruthGraph


@dataclass
class DisagreementItem:
    field_name: str
    verdicts: dict[str, str]           # annotator_id → verdict
    values: dict[str, Any]             # annotator_id → corrected_value (if any)
    resolved_by: str | None = None     # arbitrator_id if arbitrated
    resolution: Any = None


@dataclass
class ConsensusResult:
    case_id: str
    validated: bool
    validated_gtg: GroundTruthGraph
    disagreements: list[DisagreementItem] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    apply_errors: list[str] = field(default_factory=list)
    experts: list[str] = field(default_factory=list)
    arbitrator: str | None = None


def _field_consensus(
    field_name: str,
    annotations: list[GTGAnnotation],
    arbitration: GTGAnnotation | None,
) -> tuple[str, Any, DisagreementItem | None]:
    """
    Resolve one reviewable field.
    Returns (status, value, disagreement) where status ∈
    {"keep", "apply", "disagree"} and value is the correction when status=="apply".
    """
    verdicts = {a.annotator_id: a.field_judgments.get(field_name) for a in annotations}
    kinds = [j.verdict if j else Verdict.ACCEPT for j in verdicts.values()]

    if all(k == Verdict.ACCEPT for k in kinds):
        return "keep", None, None

    # Unanimous EDIT to the same corrected value?
    edits = [j for j in verdicts.values() if j and j.verdict == Verdict.EDIT]
    if len(edits) == len(annotations) and edits:
        distinct = {_freeze(j.corrected_value) for j in edits}
        if len(distinct) == 1:
            return "apply", edits[0].corrected_value, None

    # Disagreement — try arbitration.
    dis = DisagreementItem(
        field_name=field_name,
        verdicts={aid: (j.verdict.value if j else "accept") for aid, j in verdicts.items()},
        values={aid: (j.corrected_value if j else None) for aid, j in verdicts.items()},
    )
    if arbitration is not None:
        arb_j = arbitration.field_judgments.get(field_name)
        if arb_j is not None:
            dis.resolved_by = arbitration.annotator_id
            if arb_j.verdict == Verdict.EDIT:
                dis.resolution = arb_j.corrected_value
                return "apply", arb_j.corrected_value, dis
            if arb_j.verdict == Verdict.ACCEPT:
                dis.resolution = "keep_draft"
                return "keep", None, dis
            # arbitrator REJECT → still unresolved (needs a new value)
        return "disagree", None, dis
    return "disagree", None, dis


def _freeze(v: Any) -> Any:
    """Hashable representation for equality comparison of corrected values."""
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    return v


def _majority(labels: list[Any], arbitration_label: Any | None) -> tuple[Any, bool]:
    """Majority vote; returns (winner, is_tie_broken_by_arbitration)."""
    counts = Counter(labels)
    top, top_n = counts.most_common(1)[0]
    tie = sum(1 for _, n in counts.items() if n == top_n) > 1
    if tie and arbitration_label is not None:
        return arbitration_label, True
    return top, False


def merge_annotations(
    draft_gtg: GroundTruthGraph,
    annotations: list[GTGAnnotation],
    arbitration: GTGAnnotation | None = None,
) -> ConsensusResult:
    """Merge ≥2 expert annotations (+ optional arbitrator) into a validated GTG."""
    if len(annotations) < 2:
        raise ValueError("consensus requires ≥2 expert annotations")

    updates: dict[str, Any] = {}
    disagreements: list[DisagreementItem] = []
    unresolved: list[str] = []
    apply_errors: list[str] = []

    # --- reviewable content fields ---
    for fname in REVIEWABLE_FIELDS:
        status, value, dis = _field_consensus(fname, annotations, arbitration)
        if dis:
            disagreements.append(dis)
        if status == "apply":
            updates[fname] = value
        elif status == "disagree":
            unresolved.append(fname)

    # --- final_dx (special: uses final_dx_agree + final_dx_corrected) ---
    if all(a.final_dx_agree for a in annotations):
        pass  # keep draft final_dx
    else:
        corrected = {_freeze(a.final_dx_corrected) for a in annotations
                     if not a.final_dx_agree and a.final_dx_corrected}
        if len(corrected) == 1:
            updates["final_dx"] = next(a.final_dx_corrected for a in annotations
                                       if not a.final_dx_agree and a.final_dx_corrected)
        elif arbitration is not None and arbitration.final_dx_corrected:
            updates["final_dx"] = arbitration.final_dx_corrected
        else:
            unresolved.append("final_dx")
            disagreements.append(DisagreementItem(
                field_name="final_dx",
                verdicts={a.annotator_id: ("agree" if a.final_dx_agree else "disagree")
                          for a in annotations},
                values={a.annotator_id: a.final_dx_corrected for a in annotations},
            ))

    # --- stratification labels (majority vote) ---
    arb_strata = arbitration.strata if arbitration else None
    diff, _ = _majority([a.strata.difficulty for a in annotations],
                        arb_strata.difficulty if arb_strata else None)
    rar, _ = _majority([a.strata.rarity for a in annotations],
                       arb_strata.rarity if arb_strata else None)
    ep, _ = _majority([a.strata.error_prone for a in annotations],
                      arb_strata.error_prone if arb_strata else None)
    updates["difficulty"] = diff
    updates["rarity"] = rar
    updates["error_prone"] = ep

    # --- overall validity consensus ---
    valid_votes = [a.overall_valid for a in annotations]
    overall_valid, _ = _majority(valid_votes,
                                 arbitration.overall_valid if arbitration else None)

    # --- provenance ---
    experts = [a.annotator_id for a in annotations]
    updates["expert_1"] = experts[0]
    updates["expert_2"] = experts[1] if len(experts) > 1 else ""
    updates["arbitrator"] = arbitration.annotator_id if arbitration else None

    validated = (not unresolved) and bool(overall_valid)
    updates["validated"] = validated

    # --- apply updates, catching type errors per-field so one bad field doesn't
    #     sink the whole merge ---
    gtg = draft_gtg
    for k, v in updates.items():
        try:
            gtg = gtg.model_copy(update={k: v})
        except Exception as e:
            apply_errors.append(f"{k}: {e}")

    return ConsensusResult(
        case_id=draft_gtg.case_id,
        validated=validated and not apply_errors,
        validated_gtg=gtg,
        disagreements=disagreements,
        unresolved=unresolved,
        apply_errors=apply_errors,
        experts=experts,
        arbitrator=arbitration.annotator_id if arbitration else None,
    )
