"""
Bayesian LR-chain reasoning metrics (§2.1).

Clinical hypothesis-deductive reasoning is scored by whether the model's stated
probability for a diagnosis moves *consistently with* the likelihood ratios (LR)
of the findings it uncovers, and whether it makes the correct test-vs-treat
decision once its posterior crosses the case's thresholds.

Math (standard odds form; assumes conditional independence of findings — the
usual simplifying assumption, stated explicitly):

    pre_odds  = p / (1 - p)
    post_odds = pre_odds * Π LR_i
    post_p    = post_odds / (1 + post_odds)

The "Bayesian consistency" score compares, for each interval between two of the
model's differential submissions, the *direction and magnitude* of the model's
probability change against the LR-implied change, in log-odds space. It rewards
moving the right way by roughly the right amount and penalises anchoring
(not moving when strong evidence arrived) and over-/counter-updating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_EPS = 1e-6


def _clamp_p(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


def logit(p: float) -> float:
    p = _clamp_p(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def apply_lr(prior_p: float, lrs: list[float]) -> float:
    """Sequentially apply likelihood ratios to a prior probability."""
    log_odds = logit(prior_p)
    for lr in lrs:
        if lr and lr > 0:
            log_odds += math.log(lr)
    return sigmoid(log_odds)


def posterior_from_prior(prior_p: float, lr: float) -> float:
    """Single-finding posterior (convenience)."""
    return apply_lr(prior_p, [lr])


# ---------------------------------------------------------------------------
# Extracting the model's reasoning trajectory from an encounter trace
# ---------------------------------------------------------------------------

@dataclass
class DiffSnapshot:
    turn: int
    probs: dict[str, float]           # dx → stated probability


@dataclass
class FindingEvent:
    turn: int
    lr_by_dx: dict[str, float]        # dx → LR contributed by this finding


@dataclass
class BayesianTrace:
    snapshots: list[DiffSnapshot] = field(default_factory=list)
    findings: list[FindingEvent] = field(default_factory=list)

    @classmethod
    def from_trace(cls, trace: dict) -> "BayesianTrace":
        snaps: list[DiffSnapshot] = []
        finds: list[FindingEvent] = []
        for t in trace.get("turns", []):
            turn = t.get("turn", 0)
            action = t.get("action", {})
            obs = t.get("observation", {})

            if action.get("action") == "submit_differential":
                ranked = action.get("params", {}).get("ranked", [])
                probs = {e["dx"]: float(e.get("p", 0.0)) for e in ranked if "dx" in e}
                if probs:
                    snaps.append(DiffSnapshot(turn=turn, probs=probs))

            # LR-bearing findings arrive via the previous action's result,
            # surfaced in the *next* observation's last_action_result.
            lar = obs.get("channels", {}).get("last_action_result") or {}
            lr_pairs = lar.get("lr_pairs") or []
            merged: dict[str, float] = {}
            for pair in lr_pairs:
                for dx, lr in pair.items():
                    merged[dx] = float(lr)
            if merged:
                finds.append(FindingEvent(turn=turn, lr_by_dx=merged))
        return cls(snapshots=snaps, findings=finds)


def score_bayesian_consistency(bt: BayesianTrace) -> float | None:
    """
    Score in [0,1] of how Bayesian-consistent the model's probability updates are.
    Returns None when there is insufficient data (< 2 differential snapshots).

    For each consecutive snapshot pair and each dx present in both, we compute:
      expected Δlog-odds = Σ log(LR) for findings revealed in the interval
      actual   Δlog-odds = logit(p_new) - logit(p_old)
    Score per dx = f(actual, expected):
      - if expected≈0 (no evidence): reward small |actual| (no spurious update)
      - else: reward same sign and comparable magnitude (ratio-based)
    """
    snaps = bt.snapshots
    if len(snaps) < 2:
        return None

    per_dx_scores: list[float] = []
    for a, b in zip(snaps, snaps[1:]):
        interval_findings = [
            f for f in bt.findings if a.turn < f.turn <= b.turn
        ]
        common = set(a.probs) & set(b.probs)
        for dx in common:
            expected_delta = sum(
                math.log(f.lr_by_dx[dx])
                for f in interval_findings
                if dx in f.lr_by_dx and f.lr_by_dx[dx] > 0
            )
            actual_delta = logit(b.probs[dx]) - logit(a.probs[dx])
            per_dx_scores.append(_pair_score(actual_delta, expected_delta))

    if not per_dx_scores:
        return None
    return sum(per_dx_scores) / len(per_dx_scores)


def _pair_score(actual: float, expected: float, tol: float = 0.5) -> float:
    """
    Compare actual vs expected log-odds change for one diagnosis → [0,1].

    When no LR evidence bears on this dx in the interval (|expected| < tol) we
    penalise only an unsupported *increase* in probability (fabricated
    confidence). A decrease is typically benign differential re-normalisation as
    a competing dx rises, so it is not penalised here — dangerous premature
    dismissal of a must-not-miss dx is caught by the error-taxonomy layer.
    """
    if abs(expected) < tol:
        return max(0.0, 1.0 - max(0.0, actual) / (2.0 * tol + _EPS))
    if actual == 0.0:
        return 0.0  # anchoring: strong evidence but no update
    if (actual > 0) != (expected > 0):
        return 0.0  # updated the wrong direction
    # Same direction: reward magnitude agreement (log-ratio closeness).
    ratio = actual / expected
    # ratio 1 → perfect; penalise both under- (ratio<1) and over- (ratio>1) update.
    return max(0.0, 1.0 - abs(math.log(max(ratio, _EPS))))


# ---------------------------------------------------------------------------
# Threshold decision model (§2.1 test-vs-treat-vs-defer)
# ---------------------------------------------------------------------------

@dataclass
class ThresholdDecision:
    posterior: float
    test_threshold: float
    treatment_threshold: float
    correct_action: str               # "defer" | "test" | "treat"
    model_action: str | None = None
    correct: bool = False


def correct_threshold_action(
    posterior: float, test_threshold: float, treatment_threshold: float
) -> str:
    if posterior < test_threshold:
        return "defer"
    if posterior >= treatment_threshold:
        return "treat"
    return "test"


def score_threshold_decision(
    posterior: float,
    test_threshold: float,
    treatment_threshold: float,
    model_action: str | None,
) -> ThresholdDecision:
    """
    Map the model's decision verb to the correct threshold action.
    model_action is normalised from the `choose_next_step` decision / final action.
    """
    correct = correct_threshold_action(posterior, test_threshold, treatment_threshold)
    normalised = _normalise_action(model_action)
    return ThresholdDecision(
        posterior=posterior,
        test_threshold=test_threshold,
        treatment_threshold=treatment_threshold,
        correct_action=correct,
        model_action=normalised,
        correct=(normalised == correct),
    )


def _normalise_action(action: str | None) -> str | None:
    if not action:
        return None
    a = action.lower()
    if any(k in a for k in ("treat", "治疗", "处置", "prescribe", "pci", "手术", "admit")):
        return "treat"
    if any(k in a for k in ("test", "检查", "order", "workup", "imaging", "化验")):
        return "test"
    if any(k in a for k in ("defer", "discharge", "观察", "随访", "reassure", "rule_out", "排除")):
        return "defer"
    return None


def leading_posterior(bt: BayesianTrace) -> tuple[str, float] | None:
    """Return the (dx, prob) with the highest probability in the last snapshot."""
    if not bt.snapshots:
        return None
    last = bt.snapshots[-1]
    if not last.probs:
        return None
    dx = max(last.probs, key=lambda k: last.probs[k])
    return dx, last.probs[dx]
