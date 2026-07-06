"""
Cognitive error taxonomy (§2.2) — DEER / Graber-style failure analysis.

When an encounter reaches the WRONG final diagnosis, classify the likely
cognitive error(s) so the benchmark produces an *error spectrum*, not just an
accuracy number. Each error type has a rule-based signal computed from the
encounter trace + GTG; an optional LLM pass can refine ambiguous cases.

Error types (Graber cognitive dispositions to respond):
- anchoring:          fixed on the initial impression despite disconfirming evidence
- premature_closure:  stopped reasoning after the first plausible dx
- confirmation_bias:  sought only confirming tests, ignored refuting ones
- availability:       favoured a recently/easily recalled dx over base rates
- base_rate_neglect:  chose a rarer dx over a much more probable one on similar evidence
- search_satisficing: stopped after one abnormality, missed a must-not-miss dx
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from clinicraft.metrics.bayesian import BayesianTrace


class CognitiveError(str, Enum):
    NO_ERROR = "no_error"
    ANCHORING = "anchoring"
    PREMATURE_CLOSURE = "premature_closure"
    CONFIRMATION_BIAS = "confirmation_bias"
    AVAILABILITY = "availability"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    SEARCH_SATISFICING = "search_satisficing"


@dataclass
class ErrorFinding:
    error: CognitiveError
    signal_strength: float            # [0,1] confidence in the rule signal
    evidence: str


@dataclass
class ErrorReport:
    diagnosis_correct: bool
    errors: list[ErrorFinding] = field(default_factory=list)

    def primary(self) -> CognitiveError:
        if self.diagnosis_correct or not self.errors:
            return CognitiveError.NO_ERROR
        return max(self.errors, key=lambda e: e.signal_strength).error


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "")


def classify_cognitive_errors(
    trace: dict,
    gtg: dict,
) -> ErrorReport:
    """
    Rule-based cognitive-error classification for a single encounter.
    `gtg` is the ground_truth_graph dict (final_dx, differential, must_not_miss).
    """
    final_dx = _extract_final_dx(trace)
    true_dx = _norm(gtg.get("final_dx", ""))
    correct = bool(final_dx) and _norm(final_dx) == true_dx

    report = ErrorReport(diagnosis_correct=correct)
    if correct:
        return report

    turns = trace.get("turns", [])
    diffs = _differential_snapshots(turns)
    bt = BayesianTrace.from_trace(trace)

    # --- anchoring: first-impression dx == final dx, and disconfirming findings
    #     arrived but the leading probability never dropped materially.
    if diffs:
        first_top = max(diffs[0], key=lambda e: e.get("p", 0)).get("dx") if diffs[0] else None
        last_top = max(diffs[-1], key=lambda e: e.get("p", 0)).get("dx") if diffs[-1] else None
        if first_top and last_top and _norm(first_top) == _norm(last_top):
            # disconfirming evidence = any finding with LR<1 for that dx
            disconfirming = any(
                lr < 1.0
                for f in bt.findings
                for d, lr in f.lr_by_dx.items()
                if _norm(d) == _norm(first_top)
            )
            if disconfirming:
                report.errors.append(ErrorFinding(
                    CognitiveError.ANCHORING, 0.8,
                    f"首诊即锁定「{first_top}」，出现反证LR<1后仍未下调主诊断",
                ))

    # --- premature closure: very few differentials + short encounter before dx
    n_actions = len(turns)
    max_diff_breadth = max((len(d) for d in diffs), default=0)
    if max_diff_breadth <= 2 and n_actions <= 5:
        report.errors.append(ErrorFinding(
            CognitiveError.PREMATURE_CLOSURE,
            0.6 + 0.1 * (2 - max_diff_breadth),
            f"仅探索{max_diff_breadth}个鉴别、{n_actions}步即下诊断",
        ))

    # --- search satisficing: missed a must-not-miss dx entirely
    must_not_miss = {_norm(x) for x in gtg.get("must_not_miss", [])}
    mentioned = {_norm(e.get("dx", "")) for d in diffs for e in d}
    missed_mnm = must_not_miss - mentioned
    if missed_mnm:
        report.errors.append(ErrorFinding(
            CognitiveError.SEARCH_SATISFICING, 0.7,
            f"未纳入must-not-miss诊断: {', '.join(sorted(missed_mnm))}",
        ))

    # --- confirmation bias: ordered tests only for the leading dx, none that
    #     would refute it (proxy: all ordered tests appear in leading dx's workup)
    if _confirmation_signal(trace, gtg):
        report.errors.append(ErrorFinding(
            CognitiveError.CONFIRMATION_BIAS, 0.5,
            "开单检查集中于验证首选诊断，缺少可证伪的检查",
        ))

    # --- base-rate neglect: final dx is rare while a common differential had
    #     comparable/greater prior and was available.
    if _base_rate_signal(final_dx, gtg):
        report.errors.append(ErrorFinding(
            CognitiveError.BASE_RATE_NEGLECT, 0.5,
            "选择了先验概率更低的罕见诊断而非高先验的常见诊断",
        ))

    if not report.errors:
        # Wrong but no specific signal → generic no-fault/unclassified.
        report.errors.append(ErrorFinding(
            CognitiveError.NO_ERROR, 0.2, "诊断错误但无明确认知偏倚信号",
        ))
    return report


def _extract_final_dx(trace: dict) -> str | None:
    sub = trace.get("final_submission") or {}
    if "submit_diagnosis" in sub:
        return sub["submit_diagnosis"].get("dx")
    for t in reversed(trace.get("turns", [])):
        a = t.get("action", {})
        if a.get("action") == "submit_diagnosis":
            return a.get("params", {}).get("dx")
    return None


def _differential_snapshots(turns: list) -> list[list[dict]]:
    out = []
    for t in turns:
        a = t.get("action", {})
        if a.get("action") == "submit_differential":
            ranked = a.get("params", {}).get("ranked", [])
            if ranked:
                out.append(ranked)
    return out


def _confirmation_signal(trace: dict, gtg: dict) -> bool:
    ordered = [
        t.get("action", {}).get("params", {}).get("test")
        or t.get("action", {}).get("params", {}).get("study")
        for t in trace.get("turns", [])
        if t.get("action", {}).get("action") in ("order_test", "order_imaging")
    ]
    ordered = [o for o in ordered if o]
    if len(ordered) < 2:
        return False
    # If the model ordered ≥2 tests but explored ≤1 differential, that's a
    # confirmation-seeking pattern.
    diffs = _differential_snapshots(trace.get("turns", []))
    breadth = max((len(d) for d in diffs), default=0)
    return breadth <= 1


def _base_rate_signal(final_dx: str | None, gtg: dict) -> bool:
    if not final_dx:
        return False
    diff = {(_norm(e.get("dx", ""))): e.get("p_prior", e.get("p", 0.0))
            for e in gtg.get("differential", [])}
    fp = diff.get(_norm(final_dx))
    if fp is None:
        return False
    # A much more probable differential existed
    return any(p >= fp * 2 and p > 0.2 for p in diff.values())
