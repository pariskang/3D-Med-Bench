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
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_excerpt = _build_trace_excerpt(trace)

    verdict = JudgeVerdict(
        case_id=rubric.case_id,
        model_id=model_id,
        judge_model=judge_model,
    )

    # Completeness check
    verdict.completeness_ok = _check_completeness(trace, rubric)

    import asyncio
    from clinicraft.judge.multimodal_judge import judge_visual_perception

    frames = _extract_frames(trace)

    tasks = []
    for req in rubric.requirements:
        if req.auto:
            # Auto-scored items are appended immediately; they can still veto.
            verdict.requirement_scores.append(_auto_score(req, trace))
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


def _auto_score(req: RubricRequirement, trace: dict) -> RequirementScore:
    """Deterministic scoring for auto=True requirements."""
    score = 0.0
    rationale = "自动评分"

    if "贝叶斯" in req.description:
        score = _check_bayesian_update(trace)
        rationale = f"贝叶斯更新一致性分析: {score}"
    elif "成本" in req.description or "Choosing" in req.description:
        score = _check_overtesting(trace)
        rationale = f"过度检查检测: {score}"
    elif "ECE" in req.description:
        score = _check_calibration(trace)
        rationale = f"校准分析: {score}"
    else:
        # Default: check if action type appears in trace
        key = req.description[:20]
        for t in trace.get("turns", []):
            if key in json.dumps(t.get("action", {}), ensure_ascii=False):
                score = 1.0
                break

    return RequirementScore(req_id=req.id, score=score, rationale=rationale)


def _check_bayesian_update(trace: dict) -> float:
    """Check if differential probabilities increase after positive findings (simplified)."""
    diffs = []
    for t in trace.get("turns", []):
        action = t.get("action", {})
        if action.get("action") == "submit_differential":
            diffs.append(action.get("params", {}).get("ranked", []))
    if len(diffs) < 2:
        return 0.5
    # Check if top diagnosis probability is non-decreasing after findings
    tops = []
    for d in diffs:
        if d:
            tops.append(max(e.get("p", 0) for e in d))
    if len(tops) >= 2 and tops[-1] >= tops[0]:
        return 1.0
    return 0.3


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


def _check_calibration(trace: dict) -> float:
    confidences = []
    for t in trace.get("turns", []):
        a = t.get("action", {})
        if a.get("action") == "submit_diagnosis":
            c = a.get("params", {}).get("confidence")
            if c is not None:
                confidences.append(float(c))
    if not confidences:
        return 0.5
    avg_conf = sum(confidences) / len(confidences)
    return 1.0 if 0.5 <= avg_conf <= 0.95 else 0.3


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
