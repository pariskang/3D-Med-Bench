"""
Inter-rater reliability (IRR) — κ statistics, pure Python (no scipy).

- cohen_kappa:          2 raters, nominal categories
- weighted_cohen_kappa: 2 raters, ordinal categories (linear or quadratic weights)
- fleiss_kappa:         ≥3 raters, fixed ratings-per-item, nominal
- interpret_kappa:      Landis & Koch agreement bands
- compute_irr:          aggregate κ across the IRR_VARIABLES of a set of annotations

All formulas are the standard textbook definitions; unit tests pin them to
hand-computed values.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Cohen's kappa (2 raters)
# ---------------------------------------------------------------------------

def cohen_kappa(r1: Sequence[Any], r2: Sequence[Any]) -> float:
    """
    Cohen's κ for two raters over paired categorical labels.
    κ = (p_o − p_e) / (1 − p_e). Returns 1.0 for perfect agreement,
    and (by convention) 1.0 when both raters gave a single identical label to
    everything (p_e == 1 handled explicitly).
    """
    if len(r1) != len(r2):
        raise ValueError("rater label lists must be equal length")
    n = len(r1)
    if n == 0:
        return 0.0

    p_o = sum(1 for a, b in zip(r1, r2) if a == b) / n

    c1 = Counter(r1)
    c2 = Counter(r2)
    categories = set(c1) | set(c2)
    p_e = sum((c1.get(k, 0) / n) * (c2.get(k, 0) / n) for k in categories)

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Weighted Cohen's kappa (2 raters, ordinal)
# ---------------------------------------------------------------------------

def weighted_cohen_kappa(
    r1: Sequence[Any],
    r2: Sequence[Any],
    categories: list[Any],
    weights: str = "quadratic",
) -> float:
    """
    Weighted κ for ordinal categories.
    Disagreement weights: linear w_ij=|i−j|/(k−1); quadratic w_ij=(i−j)²/(k−1)².
    κ = 1 − Σ w_ij O_ij / Σ w_ij E_ij   (agreement cells have weight 0).
    """
    if len(r1) != len(r2):
        raise ValueError("rater label lists must be equal length")
    n = len(r1)
    k = len(categories)
    if n == 0 or k < 2:
        return 0.0

    idx = {c: i for i, c in enumerate(categories)}

    # Weight matrix
    def w(i: int, j: int) -> float:
        d = abs(i - j)
        if weights == "linear":
            return d / (k - 1)
        return (d ** 2) / ((k - 1) ** 2)

    # Observed confusion (proportions)
    O = [[0.0] * k for _ in range(k)]
    for a, b in zip(r1, r2):
        O[idx[a]][idx[b]] += 1.0 / n

    # Expected (outer product of marginals)
    c1 = Counter(r1)
    c2 = Counter(r2)
    p1 = [c1.get(c, 0) / n for c in categories]
    p2 = [c2.get(c, 0) / n for c in categories]
    E = [[p1[i] * p2[j] for j in range(k)] for i in range(k)]

    num = sum(w(i, j) * O[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * E[i][j] for i in range(k) for j in range(k))
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return 1.0 - num / den


# ---------------------------------------------------------------------------
# Fleiss' kappa (≥3 raters, fixed ratings per item)
# ---------------------------------------------------------------------------

def fleiss_kappa(item_ratings: Sequence[Sequence[Any]]) -> float:
    """
    Fleiss' κ. `item_ratings[i]` is the list of category labels assigned to
    item i (one per rater; every item must have the same number of raters).
    """
    n_items = len(item_ratings)
    if n_items == 0:
        return 0.0
    n_raters = len(item_ratings[0])
    if any(len(r) != n_raters for r in item_ratings):
        raise ValueError("every item must have the same number of ratings")
    if n_raters < 2:
        return 0.0

    categories = sorted({lbl for row in item_ratings for lbl in row}, key=str)
    cat_idx = {c: j for j, c in enumerate(categories)}

    # n_ij = number of raters who assigned item i to category j
    counts = [[0] * len(categories) for _ in range(n_items)]
    for i, row in enumerate(item_ratings):
        for lbl in row:
            counts[i][cat_idx[lbl]] += 1

    # Per-item agreement P_i
    P_i = []
    for i in range(n_items):
        s = sum(c * c for c in counts[i])
        P_i.append((s - n_raters) / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / n_items

    # Category marginals p_j and expected agreement P_e
    total = n_items * n_raters
    p_j = [sum(counts[i][j] for i in range(n_items)) / total for j in range(len(categories))]
    P_e = sum(p * p for p in p_j)

    if P_e == 1.0:
        return 1.0 if P_bar == 1.0 else 0.0
    return (P_bar - P_e) / (1.0 - P_e)


# ---------------------------------------------------------------------------
# Interpretation & aggregate report
# ---------------------------------------------------------------------------

def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) agreement bands."""
    if k < 0.0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost_perfect"


@dataclass
class VariableIRR:
    variable: str
    kappa: float
    method: str                 # cohen | weighted_cohen | fleiss
    n_items: int
    n_raters: int
    interpretation: str
    passes_gate: bool


@dataclass
class IRRReport:
    n_cases: int
    n_raters: int
    threshold: float
    variables: list[VariableIRR] = field(default_factory=list)
    mean_kappa: float = 0.0
    overall_pass: bool = False

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "n_raters": self.n_raters,
            "threshold": self.threshold,
            "mean_kappa": round(self.mean_kappa, 4),
            "overall_pass": self.overall_pass,
            "variables": [
                {
                    "variable": v.variable, "kappa": round(v.kappa, 4),
                    "method": v.method, "n_items": v.n_items,
                    "n_raters": v.n_raters, "interpretation": v.interpretation,
                    "passes_gate": v.passes_gate,
                }
                for v in self.variables
            ],
        }


def compute_irr(
    annotations_by_case: dict[str, list],
    threshold: float = 0.8,
) -> IRRReport:
    """
    Compute κ for each IRR variable across cases.

    `annotations_by_case[case_id]` is a list of GTGAnnotation objects for that
    case (one per rater). Every case must have the same set of raters in the
    same order. For 2 raters → Cohen (weighted for ordinal); ≥3 → Fleiss.
    """
    from clinicraft.annotation.schema import IRR_VARIABLES

    cases = [c for c in annotations_by_case.values() if c]
    if not cases:
        return IRRReport(0, 0, threshold)

    n_raters = len(cases[0])
    # Only keep cases with the full rater panel.
    complete = [c for c in cases if len(c) == n_raters]

    report = IRRReport(n_cases=len(complete), n_raters=n_raters, threshold=threshold)

    for var, (kind, categories) in IRR_VARIABLES.items():
        # Build per-item label rows: labels[item] = [rater1_label, rater2_label, ...]
        rows = [[ann.irr_labels()[var] for ann in case] for case in complete]
        if not rows:
            continue

        if n_raters == 2:
            r1 = [row[0] for row in rows]
            r2 = [row[1] for row in rows]
            if kind == "ordinal" and categories:
                kappa = weighted_cohen_kappa(r1, r2, categories, "quadratic")
                method = "weighted_cohen"
            else:
                kappa = cohen_kappa(r1, r2)
                method = "cohen"
        else:
            kappa = fleiss_kappa(rows)
            method = "fleiss"

        report.variables.append(VariableIRR(
            variable=var, kappa=kappa, method=method,
            n_items=len(rows), n_raters=n_raters,
            interpretation=interpret_kappa(kappa),
            passes_gate=kappa >= threshold,
        ))

    if report.variables:
        report.mean_kappa = sum(v.kappa for v in report.variables) / len(report.variables)
        report.overall_pass = all(v.passes_gate for v in report.variables)
    return report
