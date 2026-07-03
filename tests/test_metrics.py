"""
Tests for the §2 measurement metrics. Expected values are hand-computed so a
regression in the math is caught, not just a crash.
"""

import math
import pytest

from clinicraft.metrics.bayesian import (
    apply_lr, posterior_from_prior, BayesianTrace, score_bayesian_consistency,
    score_threshold_decision, correct_threshold_action, leading_posterior,
)
from clinicraft.metrics.calibration import expected_calibration_error, brier_score
from clinicraft.metrics.consistency import diagnostic_consistency, flip_rate, extract_final_dx
from clinicraft.metrics.error_taxonomy import classify_cognitive_errors, CognitiveError


# --------------------------------------------------------------------------
# Bayesian LR math
# --------------------------------------------------------------------------

def test_apply_lr_prior_half():
    # prior 0.5 → odds 1 → ×10 → odds 10 → p = 10/11
    assert apply_lr(0.5, [10.0]) == pytest.approx(10 / 11, abs=1e-4)


def test_apply_lr_prior_low():
    # prior 0.1 → odds 1/9 → ×10 → 10/9 → p = (10/9)/(1+10/9) = 10/19
    assert apply_lr(0.1, [10.0]) == pytest.approx(10 / 19, abs=1e-4)


def test_apply_lr_chain_multiplicative():
    # applying [4, 2.5] == applying [10]
    assert apply_lr(0.2, [4.0, 2.5]) == pytest.approx(apply_lr(0.2, [10.0]), abs=1e-6)


def test_lr_negative_lowers_probability():
    assert posterior_from_prior(0.5, 0.1) < 0.5


def test_bayesian_consistency_rewards_correct_update():
    """Model raises P(STEMI) 0.3→0.9 after an LR=100 finding → high consistency."""
    # expected Δlog-odds = log(100) ≈ 4.6; actual = logit(0.9)-logit(0.3) ≈ 2.2+0.85=... check sign/mag
    trace = {"turns": [
        {"turn": 1, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "STEMI", "p": 0.3}]}},
         "observation": {"channels": {}}},
        {"turn": 2, "action": {"action": "order_test", "params": {}},
         "observation": {"channels": {"last_action_result": {"lr_pairs": [{"STEMI": 100.0}]}}}},
        {"turn": 3, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "STEMI", "p": 0.95}]}},
         "observation": {"channels": {}}},
    ]}
    bt = BayesianTrace.from_trace(trace)
    assert len(bt.snapshots) == 2
    assert len(bt.findings) == 1
    score = score_bayesian_consistency(bt)
    assert score is not None and score > 0.5


def test_bayesian_consistency_penalises_anchoring():
    """Model keeps P(STEMI)=0.3 despite an LR=100 finding → anchoring, low score."""
    trace = {"turns": [
        {"turn": 1, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "STEMI", "p": 0.3}]}},
         "observation": {"channels": {}}},
        {"turn": 2, "action": {"action": "order_test", "params": {}},
         "observation": {"channels": {"last_action_result": {"lr_pairs": [{"STEMI": 100.0}]}}}},
        {"turn": 3, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "STEMI", "p": 0.30}]}},
         "observation": {"channels": {}}},
    ]}
    score = score_bayesian_consistency(BayesianTrace.from_trace(trace))
    assert score is not None and score < 0.3


def test_bayesian_consistency_ignores_normalisation_drop():
    """Dropping a competitor dx (no LR evidence) while the lead rises is benign."""
    trace = {"turns": [
        {"turn": 1, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "STEMI", "p": 0.3}, {"dx": "夹层", "p": 0.2}]}},
         "observation": {"channels": {}}},
        {"turn": 2, "action": {"action": "order_test", "params": {}},
         "observation": {"channels": {"last_action_result": {"lr_pairs": [{"STEMI": 20.0}]}}}},
        {"turn": 3, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "STEMI", "p": 0.9}, {"dx": "夹层", "p": 0.05}]}},
         "observation": {"channels": {}}},
    ]}
    score = score_bayesian_consistency(BayesianTrace.from_trace(trace))
    # STEMI update is near-perfect; 夹层 drop must not drag the score down.
    assert score is not None and score > 0.8


def test_bayesian_consistency_penalises_unsupported_increase():
    """Raising a dx with no supporting LR evidence is fabricated confidence."""
    trace = {"turns": [
        {"turn": 1, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "夹层", "p": 0.1}]}}, "observation": {"channels": {}}},
        {"turn": 2, "action": {"action": "ask", "params": {}},
         "observation": {"channels": {"last_action_result": {}}}},
        {"turn": 3, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "夹层", "p": 0.8}]}}, "observation": {"channels": {}}},
    ]}
    score = score_bayesian_consistency(BayesianTrace.from_trace(trace))
    assert score is not None and score < 0.5


def test_bayesian_consistency_insufficient_data():
    trace = {"turns": [{"turn": 1, "action": {"action": "submit_differential",
        "params": {"ranked": [{"dx": "X", "p": 0.5}]}}, "observation": {"channels": {}}}]}
    assert score_bayesian_consistency(BayesianTrace.from_trace(trace)) is None


# --------------------------------------------------------------------------
# Threshold decision (§2.1)
# --------------------------------------------------------------------------

def test_threshold_action_zones():
    assert correct_threshold_action(0.02, 0.05, 0.70) == "defer"
    assert correct_threshold_action(0.40, 0.05, 0.70) == "test"
    assert correct_threshold_action(0.85, 0.05, 0.70) == "treat"


def test_threshold_decision_correct_treat():
    dec = score_threshold_decision(0.85, 0.05, 0.70, "启动PCI治疗")
    assert dec.correct_action == "treat"
    assert dec.model_action == "treat"
    assert dec.correct is True


def test_threshold_decision_wrong():
    dec = score_threshold_decision(0.85, 0.05, 0.70, "继续观察随访")
    assert dec.correct_action == "treat"
    assert dec.correct is False


# --------------------------------------------------------------------------
# Calibration (ECE / Brier)
# --------------------------------------------------------------------------

def test_ece_overconfident():
    # 10 preds at conf 0.9, half correct → ECE = |0.5 - 0.9| = 0.4
    pairs = [(0.9, True)] * 5 + [(0.9, False)] * 5
    rep = expected_calibration_error(pairs, n_bins=10)
    assert rep.ece == pytest.approx(0.4, abs=1e-6)
    assert rep.overconfidence == pytest.approx(0.4, abs=1e-6)


def test_ece_perfect_calibration():
    # conf 0.0 all wrong + conf 1.0 all right → ECE 0
    pairs = [(1.0, True)] * 5 + [(0.0, False)] * 5
    rep = expected_calibration_error(pairs, n_bins=10)
    assert rep.ece == pytest.approx(0.0, abs=1e-6)
    assert rep.c6_score() == pytest.approx(100.0, abs=1e-6)


def test_brier_score_value():
    pairs = [(0.9, True)] * 5 + [(0.9, False)] * 5
    # (0.9-1)^2=0.01 ×5, (0.9-0)^2=0.81 ×5 → mean 0.41
    assert brier_score(pairs) == pytest.approx(0.41, abs=1e-6)


# --------------------------------------------------------------------------
# Consistency (§2.5)
# --------------------------------------------------------------------------

def test_consistency_all_agree():
    rep = diagnostic_consistency(["STEMI"] * 10)
    assert rep.modal_agreement == 1.0
    assert rep.normalised_entropy == 0.0
    assert rep.consistency_score == 100.0


def test_consistency_scattered():
    rep = diagnostic_consistency(["A", "B", "C", "D"])
    assert rep.modal_agreement == 0.25
    assert rep.normalised_entropy == pytest.approx(1.0, abs=1e-6)
    assert rep.consistency_score < 30


def test_consistency_modal():
    rep = diagnostic_consistency(["STEMI", "STEMI", "STEMI", "NSTEMI"])
    assert rep.modal_dx == "stemi"
    assert rep.modal_agreement == 0.75


def test_flip_rate():
    assert flip_rate(["A", "A", "B", "B"]) == pytest.approx(1 / 3, abs=1e-6)


# --------------------------------------------------------------------------
# Cognitive error taxonomy (§2.2)
# --------------------------------------------------------------------------

def test_correct_dx_no_error():
    trace = {"final_submission": {"submit_diagnosis": {"dx": "STEMI"}}, "turns": []}
    gtg = {"final_dx": "STEMI"}
    rep = classify_cognitive_errors(trace, gtg)
    assert rep.diagnosis_correct is True
    assert rep.primary() == CognitiveError.NO_ERROR


def test_search_satisficing_missed_mnm():
    trace = {"final_submission": {"submit_diagnosis": {"dx": "胃炎"}}, "turns": [
        {"turn": 1, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "胃炎", "p": 0.8}]}}},
    ]}
    gtg = {"final_dx": "STEMI", "must_not_miss": ["主动脉夹层"], "differential": []}
    rep = classify_cognitive_errors(trace, gtg)
    assert rep.diagnosis_correct is False
    errs = {e.error for e in rep.errors}
    assert CognitiveError.SEARCH_SATISFICING in errs


def test_premature_closure():
    trace = {"final_submission": {"submit_diagnosis": {"dx": "感冒"}}, "turns": [
        {"turn": 1, "action": {"action": "submit_differential",
            "params": {"ranked": [{"dx": "感冒", "p": 0.9}]}}},
        {"turn": 2, "action": {"action": "submit_diagnosis", "params": {"dx": "感冒"}}},
    ]}
    gtg = {"final_dx": "脑膜炎", "must_not_miss": [], "differential": []}
    rep = classify_cognitive_errors(trace, gtg)
    errs = {e.error for e in rep.errors}
    assert CognitiveError.PREMATURE_CLOSURE in errs
