"""
Calibration metrics (§2.5 / C6).

Real Expected Calibration Error (ECE), Brier score, and directional
over/under-confidence, computed across a set of (confidence, correct) pairs
collected from many encounters. Replaces the previous "avg confidence in
[0.5, 0.95]" heuristic.

ECE (M-bin, equal-width):
    ECE = Σ_m (|B_m|/N) · |acc(B_m) − conf(B_m)|
where B_m is the set of predictions whose confidence falls in bin m.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationReport:
    n: int
    ece: float
    brier: float
    mean_confidence: float
    accuracy: float
    overconfidence: float             # mean_conf − accuracy (>0 = overconfident)
    bins: list[dict] = field(default_factory=list)

    def c6_score(self) -> float:
        """Map ECE to a 0-100 C6 score (ECE 0 → 100, ECE ≥ 0.5 → 0)."""
        return max(0.0, min(100.0, (1.0 - self.ece / 0.5) * 100.0))


def brier_score(pairs: list[tuple[float, bool]]) -> float:
    """Mean squared error between confidence and outcome. Lower is better."""
    if not pairs:
        return 0.0
    return sum((conf - (1.0 if correct else 0.0)) ** 2 for conf, correct in pairs) / len(pairs)


def expected_calibration_error(
    pairs: list[tuple[float, bool]], n_bins: int = 10
) -> CalibrationReport:
    """
    Compute ECE + Brier + over/under-confidence from (confidence, correct) pairs.
    Confidence values are clamped to [0,1].
    """
    pairs = [(min(1.0, max(0.0, c)), bool(ok)) for c, ok in pairs]
    n = len(pairs)
    if n == 0:
        return CalibrationReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    bins: list[dict] = []
    ece = 0.0
    for m in range(n_bins):
        lo = m / n_bins
        hi = (m + 1) / n_bins
        # last bin is inclusive of 1.0
        in_bin = [
            (c, ok) for c, ok in pairs
            if (c >= lo and c < hi) or (m == n_bins - 1 and c == 1.0)
        ]
        if not in_bin:
            bins.append({"lo": lo, "hi": hi, "n": 0, "conf": 0.0, "acc": 0.0})
            continue
        conf = sum(c for c, _ in in_bin) / len(in_bin)
        acc = sum(1 for _, ok in in_bin if ok) / len(in_bin)
        ece += (len(in_bin) / n) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "n": len(in_bin), "conf": conf, "acc": acc})

    mean_conf = sum(c for c, _ in pairs) / n
    accuracy = sum(1 for _, ok in pairs if ok) / n
    return CalibrationReport(
        n=n,
        ece=ece,
        brier=brier_score(pairs),
        mean_confidence=mean_conf,
        accuracy=accuracy,
        overconfidence=mean_conf - accuracy,
        bins=bins,
    )


def collect_confidence_pairs(scorecards: list) -> list[tuple[float, bool]]:
    """
    Build (confidence, correct) pairs from a list of scored encounters.
    Each scorecard must expose `.detail['final_confidence']` and a C1-derived
    correctness flag; falls back gracefully when absent.
    """
    pairs: list[tuple[float, bool]] = []
    for card in scorecards:
        detail = getattr(card, "detail", {}) or {}
        conf = detail.get("final_confidence")
        if conf is None:
            continue
        correct = getattr(card, "C1", 0.0) >= 50.0  # C1 pass → diagnosis correct
        pairs.append((float(conf), bool(correct)))
    return pairs
