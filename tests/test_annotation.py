"""
Tests for the expert-annotation workflow. κ values are hand-computed so a
regression in the statistics is caught, not just a crash.
"""

import pytest

from clinicraft.annotation.irr import (
    cohen_kappa, weighted_cohen_kappa, fleiss_kappa, interpret_kappa, compute_irr,
)
from clinicraft.annotation.schema import (
    GTGAnnotation, FieldJudgment, StrataJudgment, Verdict,
)
from clinicraft.annotation.consensus import merge_annotations
from clinicraft.schemas.ground_truth import GroundTruthGraph


# --------------------------------------------------------------------------
# κ statistics — hand-computed
# --------------------------------------------------------------------------

def test_cohen_kappa_known_value():
    r1 = ["V", "V", "V", "V", "V", "V", "I", "I", "I", "I"]
    r2 = ["V", "V", "V", "V", "V", "I", "I", "I", "I", "V"]
    # p_o=0.8, p_e=0.52 → κ=0.28/0.48
    assert cohen_kappa(r1, r2) == pytest.approx(0.5833, abs=1e-4)


def test_cohen_kappa_perfect():
    assert cohen_kappa(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_cohen_kappa_all_same_label():
    # both raters label everything "V" → p_e=1 → defined as 1.0 (perfect)
    assert cohen_kappa(["V"] * 5, ["V"] * 5) == 1.0


def test_weighted_kappa_quadratic_known():
    r1, r2 = [0, 1, 2], [0, 2, 2]
    kw = weighted_cohen_kappa(r1, r2, [0, 1, 2], "quadratic")
    assert kw == pytest.approx(0.8, abs=1e-6)


def test_weighted_greater_than_unweighted_for_close_disagreement():
    r1, r2 = [0, 1, 2], [0, 2, 2]
    unw = cohen_kappa(r1, r2)
    kw = weighted_cohen_kappa(r1, r2, [0, 1, 2], "quadratic")
    assert unw == pytest.approx(0.5, abs=1e-6)
    assert kw > unw  # ordinal-close disagreement penalised less


def test_fleiss_perfect_agreement():
    ratings = [["A", "A", "A"], ["B", "B", "B"]]
    assert fleiss_kappa(ratings) == pytest.approx(1.0, abs=1e-9)


def test_fleiss_worse_than_chance():
    ratings = [["A", "A", "B"], ["A", "A", "B"]]
    assert fleiss_kappa(ratings) == pytest.approx(-0.5, abs=1e-6)


def test_interpret_kappa_bands():
    assert interpret_kappa(-0.1) == "poor"
    assert interpret_kappa(0.3) == "fair"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.7) == "substantial"
    assert interpret_kappa(0.85) == "almost_perfect"


# --------------------------------------------------------------------------
# compute_irr aggregate
# --------------------------------------------------------------------------

def _ann(case_id, aid, valid=True, dx_agree=True, difficulty="hard",
         rarity="rare", error_prone=True):
    return GTGAnnotation(
        case_id=case_id, annotator_id=aid, overall_valid=valid,
        final_dx_agree=dx_agree,
        strata=StrataJudgment(difficulty=difficulty, rarity=rarity, error_prone=error_prone),
    )


def test_compute_irr_two_raters_perfect():
    by_case = {
        "c1": [_ann("c1", "a"), _ann("c1", "b")],
        "c2": [_ann("c2", "a", difficulty="easy", rarity="common"),
               _ann("c2", "b", difficulty="easy", rarity="common")],
    }
    report = compute_irr(by_case, threshold=0.8)
    assert report.n_cases == 2
    assert report.n_raters == 2
    # all variables perfectly agree → κ=1 across the board
    assert report.overall_pass is True
    assert report.mean_kappa == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# consensus / arbitration
# --------------------------------------------------------------------------

def _draft_gtg():
    return GroundTruthGraph(
        case_id="C1", problem_representation="pr", final_dx="STEMI",
        specialty="cardiology", difficulty="hard", rarity="rare",
        differential=[{"dx": "STEMI", "p_prior": 0.6}],
        must_not_miss=["夹层"], red_flags=["rf"],
        expert_reasoning_trace=["s1"],
    )


def test_consensus_all_accept_validates():
    gtg = _draft_gtg()
    a = _ann("C1", "dr_a")
    b = _ann("C1", "dr_b")
    result = merge_annotations(gtg, [a, b])
    assert result.validated is True
    assert result.validated_gtg.validated is True
    assert result.validated_gtg.expert_1 == "dr_a"
    assert result.validated_gtg.expert_2 == "dr_b"
    assert not result.unresolved


def test_consensus_unanimous_edit_applies():
    gtg = _draft_gtg()
    edit = FieldJudgment(verdict=Verdict.EDIT, corrected_value=["夹层", "肺栓塞"])
    a = _ann("C1", "dr_a"); a.field_judgments["must_not_miss"] = edit
    b = _ann("C1", "dr_b"); b.field_judgments["must_not_miss"] = edit
    result = merge_annotations(gtg, [a, b])
    assert result.validated_gtg.must_not_miss == ["夹层", "肺栓塞"]
    assert result.validated is True


def test_consensus_conflict_unresolved_without_arbitration():
    gtg = _draft_gtg()
    a = _ann("C1", "dr_a"); a.field_judgments["final_dx"] = FieldJudgment(verdict=Verdict.REJECT)
    b = _ann("C1", "dr_b")  # accepts
    result = merge_annotations(gtg, [a, b])
    assert "final_dx" in result.unresolved
    assert result.validated is False


def test_consensus_arbitration_resolves():
    gtg = _draft_gtg()
    a = _ann("C1", "dr_a"); a.field_judgments["final_dx"] = FieldJudgment(verdict=Verdict.REJECT)
    b = _ann("C1", "dr_b")
    arb = _ann("C1", "dr_arb"); arb.role = "arbitrator"
    arb.field_judgments["final_dx"] = FieldJudgment(verdict=Verdict.EDIT, corrected_value="NSTEMI")
    result = merge_annotations(gtg, [a, b], arbitration=arb)
    assert "final_dx" not in result.unresolved
    assert result.validated_gtg.final_dx == "NSTEMI"
    assert result.validated_gtg.arbitrator == "dr_arb"
    assert result.validated is True


def test_consensus_disagree_final_dx_via_dx_agree():
    gtg = _draft_gtg()
    a = _ann("C1", "dr_a", dx_agree=False); a.final_dx_corrected = "心包炎"
    b = _ann("C1", "dr_b", dx_agree=False); b.final_dx_corrected = "心包炎"
    result = merge_annotations(gtg, [a, b])
    # both independently corrected to the same dx → applied
    assert result.validated_gtg.final_dx == "心包炎"


def test_consensus_strata_majority_vote():
    gtg = _draft_gtg()
    a = _ann("C1", "a", difficulty="medium")
    b = _ann("C1", "b", difficulty="hard")
    c = _ann("C1", "c", difficulty="hard")
    result = merge_annotations(gtg, [a, b, c])
    assert result.validated_gtg.difficulty == "hard"  # 2 of 3


def test_consensus_requires_two_annotations():
    with pytest.raises(ValueError):
        merge_annotations(_draft_gtg(), [_ann("C1", "solo")])


# --------------------------------------------------------------------------
# YAML round-trip (form → parse)
# --------------------------------------------------------------------------

def test_form_roundtrip(tmp_path):
    from clinicraft.annotation.workflow import blank_form, parse_form
    import yaml
    gtg = _draft_gtg()
    form = blank_form(gtg, "dr_wang")
    p = tmp_path / "dr_wang.yaml"
    p.write_text(yaml.dump(form, allow_unicode=True, sort_keys=False), encoding="utf-8")
    ann = parse_form(p)
    assert ann.case_id == "C1"
    assert ann.annotator_id == "dr_wang"
    assert ann.strata.difficulty == "hard"
    assert ann.field_judgments["final_dx"].verdict == Verdict.ACCEPT
