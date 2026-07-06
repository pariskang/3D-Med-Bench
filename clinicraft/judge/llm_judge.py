"""
LLM Judge — evaluates encounter traces against hidden rubric.
Uses claude-opus-4-8 with structured output for each rubric requirement.

Key design choices (承 GameCraft-Bench + MedCaseReasoning):
- Hidden rubric: SUT never sees rubric.json during episode
- Expert-calibrated: judge outputs calibrated against human rater κ
- Per-requirement binary score (0/1) + brief rationale
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from clinicraft.config import settings
from clinicraft.schemas.rubric import Rubric, RubricRequirement


JUDGE_SYSTEM = """你是一位严格的临床教育专家和AI评测裁判。
你的任务是根据隐藏的评测标准，评估AI医生在真实病例问诊中的表现。
你拥有完整的病例真值图（GTG），AI医生没有看到这些真值。
对每个评测条目，给出：
1. score: 0或1（是否满足标准）
2. rationale: 1-2句判断理由（引用具体证据）
3. confidence: 0.0-1.0（你对本次判断的把握程度）
判断标准：宁可严格，不可宽松。偏倚越少越好。"""


class RequirementScore(BaseModel):
    req_id: str
    score: float      # 0.0 or 1.0 (partial credit possible for some items)
    rationale: str
    confidence: float = 1.0


class JudgeVerdict(BaseModel):
    case_id: str
    model_id: str
    requirement_scores: list[RequirementScore] = []
    completeness_ok: bool = True
    veto_triggered: list[str] = []
    cognitive_errors: list[dict] = []      # §2.2 DEER/Graber error spectrum
    metrics: dict = {}                      # bayesian/threshold/calibration detail
    judge_model: str = ""
    schema_version: str = "3.0"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def _score_requirement(
    client: anthropic.AsyncAnthropic,
    model: str,
    req: RubricRequirement,
    trace_excerpt: str,
    gtg_json: str,
) -> RequirementScore:
    tool = {
        "name": "score_requirement",
        "description": "为单个评测条目评分",
        "input_schema": RequirementScore.model_json_schema(),
    }
    response = await client.messages.create(
        model=model,
        max_tokens=512,
        system=JUDGE_SYSTEM,
        tools=[tool],
        tool_choice={"type": "tool", "name": "score_requirement"},
        messages=[{
            "role": "user",
            "content": (
                f"**评测条目** [{req.cat}] {req.description}\n"
                f"**权重**: {req.weight}\n\n"
                f"**真值图摘要**:\n{gtg_json[:2000]}\n\n"
                f"**AI医生行为记录（节选）**:\n{trace_excerpt[:3000]}\n\n"
                "请评分（0=未满足, 1=完全满足, 0.5=部分满足）："
            ),
        }],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "score_requirement":
            raw = block.input
            raw["req_id"] = req.id
            return RequirementScore.model_validate(raw)
    raise RuntimeError("Judge LLM did not emit score_requirement")


async def judge_encounter(
    trace_path: Path,
    rubric_path: Path,
    gtg_path: Path,
    model_id: str = "unknown",
    client: anthropic.AsyncAnthropic | None = None,
    judge_model: str | None = None,
) -> JudgeVerdict:
    """
    Evaluate one encounter trace against its rubric and GTG.
    Auto-scored requirements are handled inline; LLM-judge requirements
    are sent to the judge model in parallel.
    """
    client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    judge_model = judge_model or settings.judge_model

    rubric = Rubric.model_validate_json(rubric_path.read_text())
    gtg_json = gtg_path.read_text(encoding="utf-8")
    gtg_dict = json.loads(gtg_json)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_excerpt = _build_trace_excerpt(trace)

    verdict = JudgeVerdict(
        case_id=rubric.case_id,
        model_id=model_id,
        judge_model=judge_model,
    )

    # Completeness check
    verdict.completeness_ok = _check_completeness(trace, rubric)

    # §2.2 cognitive-error spectrum + §2.1 reasoning metrics (deterministic).
    verdict.cognitive_errors, verdict.metrics = _analyse_reasoning(trace, gtg_dict)

    import asyncio
    from clinicraft.judge.multimodal_judge import judge_visual_perception

    frames = _extract_frames(trace)

    tasks = []
    for req in rubric.requirements:
        if req.auto:
            # Auto-scored items are appended immediately; they can still veto.
            verdict.requirement_scores.append(_auto_score(req, trace, gtg_dict))
        elif req.judge == "multimodal":
            tasks.append(judge_visual_perception(req, trace_excerpt, frames, client))
        elif req.judge == "llm":
            tasks.append(_score_requirement(client, judge_model, req, trace_excerpt, gtg_json))
        # judge == "human"/"auto" but auto=False → deferred to human review; skip.

    judged = await asyncio.gather(*tasks, return_exceptions=True)
    for s in judged:
        if isinstance(s, RequirementScore):
            verdict.requirement_scores.append(s)
        else:
            logger.error(f"Judge task failed: {s}")

    # Veto check runs over ALL scored requirements (auto + llm + multimodal).
    for s in verdict.requirement_scores:
        if _is_veto(s, rubric):
            verdict.veto_triggered.append(s.req_id)

    return verdict


def _extract_frames(trace: dict) -> list:
    """Collect unique vision frames (data URIs / paths) from trace observations."""
    seen: list = []
    for turn in trace.get("turns", []):
        vision = turn.get("observation", {}).get("channels", {}).get("vision")
        if vision and vision.get("frames"):
            for f in vision["frames"]:
                if f not in seen:
                    seen.append(f)
    return seen[:8]  # cap to bound judge cost


def _build_trace_excerpt(trace: dict) -> str:
    """Summarise key actions from the encounter trace for the judge."""
    lines = [f"Case: {trace.get('case_id')}, Model: {trace.get('model_id')}"]
    for turn in trace.get("turns", [])[:30]:
        action = turn.get("action", {})
        atype = action.get("action", "?")
        params = action.get("params", {})
        lines.append(f"T{turn['turn']}: {atype} {json.dumps(params, ensure_ascii=False)[:100]}")
    if sub := trace.get("final_submission"):
        lines.append(f"FINAL: {json.dumps(sub, ensure_ascii=False)[:200]}")
    return "\n".join(lines)


def _auto_score(req: RubricRequirement, trace: dict, gtg: dict | None = None) -> RequirementScore:
    """Deterministic scoring for auto=True requirements, backed by real metrics."""
    gtg = gtg or {}
    score = 0.0
    rationale = "自动评分"

    if "贝叶斯" in req.description or "LR" in req.description:
        from clinicraft.metrics.bayesian import BayesianTrace, score_bayesian_consistency
        bt = BayesianTrace.from_trace(trace)
        c = score_bayesian_consistency(bt)
        score = 0.5 if c is None else c
        rationale = (
            f"贝叶斯一致性={score:.2f} (snapshots={len(bt.snapshots)}, findings={len(bt.findings)})"
        )
    elif "阈值" in req.description or "threshold" in req.description.lower():
        score, rationale = _score_threshold(trace, gtg)
    elif "成本" in req.description or "Choosing" in req.description:
        score = _check_overtesting(trace)
        rationale = f"过度检查检测={score:.2f}"
    elif "ECE" in req.description or "校准" in req.description:
        score, rationale = _score_calibration(trace, gtg)
    else:
        key = req.description[:20]
        for t in trace.get("turns", []):
            if key in json.dumps(t.get("action", {}), ensure_ascii=False):
                score = 1.0
                break

    return RequirementScore(req_id=req.id, score=score, rationale=rationale)


def _score_threshold(trace: dict, gtg: dict) -> tuple[float, str]:
    """§2.1 threshold decision: is the model's test-vs-treat action correct?"""
    from clinicraft.metrics.bayesian import (
        BayesianTrace, leading_posterior, score_threshold_decision,
    )
    bt = BayesianTrace.from_trace(trace)
    lead = leading_posterior(bt)
    if not lead:
        return 0.5, "无差异诊断概率，无法评估阈值决策"
    _dx, posterior = lead
    model_action = _extract_decision_verb(trace)
    dec = score_threshold_decision(
        posterior,
        gtg.get("test_threshold", 0.05),
        gtg.get("treatment_threshold", 0.70),
        model_action,
    )
    return (1.0 if dec.correct else 0.0), (
        f"后验={posterior:.2f} 应为「{dec.correct_action}」, 模型「{dec.model_action}」"
    )


def _score_calibration(trace: dict, gtg: dict) -> tuple[float, str]:
    """
    Per-encounter calibration: confidence should track correctness.
    Score = 1 - (confidence - correct)^2  (per-item Brier complement).
    Cross-case ECE is aggregated separately by metrics.calibration.
    """
    from clinicraft.metrics.consistency import _normalise_dx, extract_final_dx
    conf = None
    for t in reversed(trace.get("turns", [])):
        a = t.get("action", {})
        if a.get("action") == "submit_diagnosis":
            conf = a.get("params", {}).get("confidence")
            break
    if conf is None:
        return 0.5, "无置信度，无法评估校准"
    final_dx = extract_final_dx(trace)
    correct = 1.0 if (final_dx and _normalise_dx(final_dx) == _normalise_dx(gtg.get("final_dx", ""))) else 0.0
    brier_item = (float(conf) - correct) ** 2
    return max(0.0, 1.0 - brier_item), (
        f"置信度={conf}, 正确={bool(correct)}, 单例Brier={brier_item:.2f}"
    )


def _extract_decision_verb(trace: dict) -> str | None:
    """Find the model's decision (choose_next_step > terminal action)."""
    for t in trace.get("turns", []):
        a = t.get("action", {})
        if a.get("action") == "choose_next_step":
            return a.get("params", {}).get("decision")
    # fall back to terminal action type
    for t in reversed(trace.get("turns", [])):
        a = t.get("action", {})
        if a.get("action") in ("prescribe", "submit_plan", "refer", "escalate"):
            return a.get("action")
        if a.get("action") in ("order_test", "order_imaging"):
            return "test"
    return None


def _analyse_reasoning(trace: dict, gtg: dict) -> tuple[list[dict], dict]:
    """Run the deterministic §2.1/§2.2 metrics; return (error_spectrum, metrics)."""
    from clinicraft.metrics.bayesian import (
        BayesianTrace, leading_posterior, score_bayesian_consistency,
        score_threshold_decision,
    )
    from clinicraft.metrics.error_taxonomy import classify_cognitive_errors

    bt = BayesianTrace.from_trace(trace)
    bayes = score_bayesian_consistency(bt)
    metrics: dict = {
        "bayesian_consistency": bayes,
        "n_diff_snapshots": len(bt.snapshots),
        "n_lr_findings": len(bt.findings),
    }
    lead = leading_posterior(bt)
    if lead:
        dx, post = lead
        dec = score_threshold_decision(
            post, gtg.get("test_threshold", 0.05),
            gtg.get("treatment_threshold", 0.70), _extract_decision_verb(trace),
        )
        metrics["leading_dx"] = dx
        metrics["leading_posterior"] = post
        metrics["threshold_correct"] = dec.correct
        metrics["threshold_expected"] = dec.correct_action

    # confidence for cross-case ECE aggregation
    for t in reversed(trace.get("turns", [])):
        a = t.get("action", {})
        if a.get("action") == "submit_diagnosis":
            metrics["final_confidence"] = a.get("params", {}).get("confidence")
            break

    report = classify_cognitive_errors(trace, gtg)
    errors = [
        {"error": e.error.value, "strength": e.signal_strength, "evidence": e.evidence}
        for e in report.errors
    ]
    metrics["diagnosis_correct"] = report.diagnosis_correct
    metrics["primary_error"] = report.primary().value
    return errors, metrics


def _check_overtesting(trace: dict) -> float:
    tests_ordered = sum(
        1 for t in trace.get("turns", [])
        if t.get("action", {}).get("action") in ("order_test", "order_imaging")
    )
    if tests_ordered <= 5:
        return 1.0
    elif tests_ordered <= 10:
        return 0.5
    return 0.0


def _check_completeness(trace: dict, rubric: Rubric) -> bool:
    cc = rubric.completeness_check
    action_types = {t.get("action", {}).get("action") for t in trace.get("turns", [])}
    # "Taking vitals" = any examination/observation action that reads the patient
    # (vitals are always shown in structured_state, so any active exam counts).
    vitals_actions = {"inspect", "auscultate", "palpate", "percuss", "check_pulse",
                      "check_cap_refill", "observe_task", "order_test"}
    if cc.must_take_vitals and action_types.isdisjoint(vitals_actions):
        return False
    if cc.must_submit_problem_rep and "submit_problem_rep" not in action_types:
        return False
    if cc.must_submit_differential and "submit_differential" not in action_types:
        return False
    if cc.must_submit_diagnosis and "submit_diagnosis" not in action_types:
        return False
    if cc.must_provide_safety_net and "safety_net" not in action_types:
        return False
    return True


def _is_veto(score: RequirementScore, rubric: Rubric) -> bool:
    for req in rubric.requirements:
        if req.id == score.req_id and req.veto_if_fail and score.score < 0.5:
            return True
    return False
