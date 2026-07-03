"""
Consistency metrics (§2.5) — "competence without consistency".

Given N encounter traces of the *same* case (run at temperature > 0), measure
how stable the model's diagnosis is. Also supports intra-cluster consistency
(same diagnosis across a cluster of clinically similar cases).

Metrics:
- modal_agreement: fraction of runs agreeing with the most common final dx
- normalised_entropy: Shannon entropy of the dx distribution / log(k), in [0,1]
                      (0 = perfectly consistent, 1 = maximally scattered)
- flip_rate: fraction of adjacent run pairs that disagree
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ConsistencyReport:
    n_runs: int
    distinct_dx: int
    modal_dx: str | None
    modal_agreement: float            # [0,1], 1 = all runs agree
    normalised_entropy: float         # [0,1], 0 = consistent
    consistency_score: float          # [0,100] headline score
    distribution: dict[str, int] = field(default_factory=dict)


def _normalise_dx(dx: str) -> str:
    return (dx or "").strip().lower().replace(" ", "")


def diagnostic_consistency(final_dxs: list[str]) -> ConsistencyReport:
    """
    Compute consistency from a list of final diagnoses (one per run).
    """
    normed = [_normalise_dx(d) for d in final_dxs if d]
    n = len(normed)
    if n == 0:
        return ConsistencyReport(0, 0, None, 0.0, 1.0, 0.0, {})

    counts = Counter(normed)
    modal_dx, modal_count = counts.most_common(1)[0]
    modal_agreement = modal_count / n

    # Shannon entropy normalised by log(k) where k = distinct dx
    k = len(counts)
    if k <= 1:
        norm_entropy = 0.0
    else:
        probs = [c / n for c in counts.values()]
        entropy = -sum(p * math.log(p) for p in probs)
        norm_entropy = entropy / math.log(k)

    # Headline score: reward modal agreement, penalise entropy.
    consistency_score = max(0.0, min(100.0, (modal_agreement * (1.0 - norm_entropy)) * 100.0))

    # Map normalised dx back to a representative original spelling
    return ConsistencyReport(
        n_runs=n,
        distinct_dx=k,
        modal_dx=modal_dx,
        modal_agreement=modal_agreement,
        normalised_entropy=norm_entropy,
        consistency_score=consistency_score,
        distribution=dict(counts),
    )


def flip_rate(final_dxs: list[str]) -> float:
    """Fraction of adjacent run pairs whose diagnosis differs."""
    normed = [_normalise_dx(d) for d in final_dxs if d]
    if len(normed) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(normed, normed[1:]) if a != b)
    return flips / (len(normed) - 1)


def extract_final_dx(trace: dict) -> str | None:
    """Pull the submitted final diagnosis from an encounter trace."""
    sub = trace.get("final_submission") or {}
    if "submit_diagnosis" in sub:
        return sub["submit_diagnosis"].get("dx")
    # fall back to scanning turns
    for t in reversed(trace.get("turns", [])):
        a = t.get("action", {})
        if a.get("action") == "submit_diagnosis":
            return a.get("params", {}).get("dx")
    return None
