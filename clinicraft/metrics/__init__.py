"""
Quantitative clinical measurement (§2).

Real, testable implementations of the reasoning/calibration constructs that the
audit flagged as heuristic-only:

- bayesian:      LR-chain pre→post-test probability + Bayesian-update consistency
                 + threshold decision correctness (§2.1)
- calibration:   binned ECE, Brier score, over/under-confidence (§2.5 / C6)
- consistency:   run-to-run and intra-cluster diagnostic stability (§2.5)
- error_taxonomy: DEER/Graber cognitive-error classification (§2.2)

All functions are pure and unit-tested; none require an API key.
"""

from clinicraft.metrics.bayesian import (
    BayesianTrace, ThresholdDecision, apply_lr, posterior_from_prior,
    score_bayesian_consistency, score_threshold_decision,
)
from clinicraft.metrics.calibration import (
    CalibrationReport, brier_score, expected_calibration_error,
)
from clinicraft.metrics.consistency import ConsistencyReport, diagnostic_consistency
from clinicraft.metrics.error_taxonomy import CognitiveError, classify_cognitive_errors

__all__ = [
    "BayesianTrace", "ThresholdDecision", "apply_lr", "posterior_from_prior",
    "score_bayesian_consistency", "score_threshold_decision",
    "CalibrationReport", "brier_score", "expected_calibration_error",
    "ConsistencyReport", "diagnostic_consistency",
    "CognitiveError", "classify_cognitive_errors",
]
