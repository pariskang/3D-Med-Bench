"""
Multimodal Judge — evaluates C3 (3D interaction & perception) requirements.
Receives patient video frames + encounter trace, assesses whether the model
correctly perceived and reported visible signs from the avatar.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

from clinicraft.config import settings
from clinicraft.judge.llm_judge import RequirementScore
from clinicraft.schemas.rubric import RubricRequirement


MULTIMODAL_SYSTEM = """你是一位临床医学教育专家，同时负责评估AI模型对3D患者化身的感知能力。
你会收到：
1. 患者化身在就诊过程中的视频帧（截图）
2. AI医生的问诊记录
3. 评测条目（描述AI是否正确感知到某个体征）

请判断：AI医生是否从视觉通道（而非文字提示）识别出了目标体征，
并将其纳入诊断推理。
给出 score（0/0.5/1）和简要rationale。"""


def _to_image_block(frame: "str | Path") -> dict[str, Any] | None:
    """Accept a data URI, a text-frame:// stub, or a PNG file path."""
    s = str(frame)
    if s.startswith("data:image/"):
        header, _, b64 = s.partition(",")
        media = header.split(";")[0].removeprefix("data:") or "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}}
    if s.startswith("text-frame://"):
        return None  # non-visual fallback frame — nothing to show the vision model
    p = Path(s)
    if p.exists():
        data = base64.standard_b64encode(p.read_bytes()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
    return None


async def judge_visual_perception(
    req: RubricRequirement,
    trace_excerpt: str,
    frame_paths: "list[str | Path]",
    client: anthropic.AsyncAnthropic | None = None,
) -> RequirementScore:
    """
    Multimodal judge for C3 visual perception requirements.
    Sends image frames + trace to Claude vision API.
    Frames may be data URIs (from the trace) or PNG file paths.
    """
    client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Build image content blocks
    content: list[dict[str, Any]] = []
    for fp in frame_paths[:4]:  # max 4 frames per assessment
        block = _to_image_block(fp)
        if block:
            content.append(block)

    has_images = len(content) > 0
    content.append({
        "type": "text",
        "text": (
            f"**评测条目**: {req.description}\n\n"
            f"**AI医生问诊记录**:\n{trace_excerpt[:2000]}\n\n"
            "以上图像为就诊过程中患者化身的截图。\n"
            "请判断AI医生是否从视觉通道正确识别了目标体征（score: 0/0.5/1），\n"
            "并给出1-2句rationale。\n"
            "以JSON格式回复：{\"req_id\": \"...\", \"score\": 0.0, \"rationale\": \"...\", \"confidence\": 0.8}"
        ),
    })

    if not has_images:
        # No usable image frames — cannot assess visual perception.
        logger.warning(f"No image frames for multimodal judge on {req.id}")
        return RequirementScore(
            req_id=req.id, score=0.0,
            rationale="无可用视频帧，无法评估3D感知（渲染器未产生图像）",
            confidence=0.3,
        )

    response = await client.messages.create(
        model=settings.judge_model,
        max_tokens=256,
        system=MULTIMODAL_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )

    import json, re
    text = response.content[0].text.strip()
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            raw = json.loads(m.group())
            raw["req_id"] = req.id
            return RequirementScore.model_validate(raw)
    except Exception:
        pass

    return RequirementScore(
        req_id=req.id, score=0.5,
        rationale=text[:200], confidence=0.5,
    )
