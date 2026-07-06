"""
Stage 1: Ingestion & Structuring
Input:  raw case text (.txt)
Output: ClinicalCase JSON

Framework: Anthropic claude-opus-4-8 tool_use → Pydantic v2 guaranteed structure.
Why LLM-only (not cTAKES/HanLP-first): Chinese clinical text has extreme format
variance (EHR, discharge summary, case report narrative, SOAP notes). A single
powerful LLM extraction pass outperforms rule-based pipelines on recall/precision
for multi-section extraction. HanLP is used only for pre-segmentation of very long
texts to stay within the LLM context window.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from clinicraft.config import settings
from clinicraft.schemas.clinical_case import ClinicalCase


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是资深临床信息提取专家。
你的任务是从中文病例文本（可能是病案首页、出院小结、病程记录、病例报告等格式）中提取结构化临床信息。
要求：
1. 忠实于原文，不推断或补充原文中未明确提及的内容
2. 将所有诊断名称标准化（尽量给出ICD-10编码）
3. 检验指标保留原始值和单位
4. 生命体征使用标准英文键名（HR/BP/RR/SpO2/T/GCS）
5. 如原文某字段缺失，保持为null/空列表，不要填充假设值
6. extraction_confidence: 你对本次提取完整性和准确性的整体估分（0.0-1.0）"""

EXTRACTION_TOOL_DESC = "从临床病例文本提取完整结构化信息，忠实原文，不推断补充。"


def _build_extraction_tool(case_id: str, source_file: str) -> dict:
    """Build the tool_use definition from the Pydantic schema."""
    schema = ClinicalCase.model_json_schema()
    # Inject fixed values so model doesn't need to generate them
    schema["properties"]["case_id"] = {"type": "string", "default": case_id}
    schema["properties"]["source_file"] = {"type": "string", "default": source_file}
    return {
        "name": "extract_clinical_case",
        "description": EXTRACTION_TOOL_DESC,
        "input_schema": schema,
    }


def _chunk_text(text: str, max_chars: int = 80_000) -> list[str]:
    """
    Split very long texts into chunks at section boundaries.
    HanLP sentence splitting would be ideal but adds a heavy dependency;
    section-boundary heuristics work well for Chinese clinical records.
    """
    if len(text) <= max_chars:
        return [text]

    # Common Chinese clinical section headers
    section_re = re.compile(
        r"(?=(?:主诉|现病史|既往史|个人史|家族史|月经史|婚育史|过敏史|"
        r"体格检查|辅助检查|实验室检查|影像学|诊断|治疗|处置|病程记录|"
        r"入院记录|出院记录|手术记录|护理记录)[\s：:]\s*)"
    )
    parts = section_re.split(text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) > max_chars:
            if current:
                chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks or [text]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm(
    client: anthropic.AsyncAnthropic,
    model: str,
    tool: dict,
    user_message: str,
) -> dict[str, Any]:
    """Single LLM call with retry. Returns raw tool_use input dict."""
    response = await client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "extract_clinical_case"},
        messages=[{"role": "user", "content": user_message}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_clinical_case":
            return block.input  # type: ignore[return-value]
    raise RuntimeError("LLM did not emit extract_clinical_case tool call")


async def extract_from_text(
    raw_text: str,
    case_id: str,
    source_file: str,
    client: anthropic.AsyncAnthropic,
    model: str | None = None,
) -> ClinicalCase:
    """
    Main extraction function: txt → ClinicalCase.
    For long texts, extracts per-chunk then merges.
    """
    model = model or settings.llm_model
    tool = _build_extraction_tool(case_id, source_file)
    chunks = _chunk_text(raw_text)

    if len(chunks) == 1:
        user_msg = f"请从以下病例文本中提取结构化信息：\n\n{raw_text}"
        raw = await _call_llm(client, model, tool, user_msg)
    else:
        logger.info(f"[{case_id}] Long text split into {len(chunks)} chunks")
        # First chunk: full extraction
        user_msg = (
            f"以下是病例第1/{len(chunks)}部分，请尽量提取所有字段：\n\n{chunks[0]}"
        )
        raw = await _call_llm(client, model, tool, user_msg)
        # Subsequent chunks: supplement/overwrite where richer
        for i, chunk in enumerate(chunks[1:], 2):
            user_msg = (
                f"以下是同一病例第{i}/{len(chunks)}部分，补充或更新前面提取的字段：\n\n{chunk}"
            )
            extra = await _call_llm(client, model, tool, user_msg)
            raw = _merge_extractions(raw, extra)

    # Inject fixed fields
    raw["case_id"] = case_id
    raw["source_file"] = source_file
    raw["source_hash"] = hashlib.sha256(raw_text.encode()).hexdigest()[:16]
    raw["extraction_model"] = model

    return ClinicalCase.model_validate(raw)


def _merge_extractions(base: dict, extra: dict) -> dict:
    """
    Merge two extraction dicts: lists are extended (deduped by str repr),
    non-null scalars from extra overwrite base only if base is null/empty.
    """
    merged = dict(base)
    for key, val in extra.items():
        if key in ("case_id", "source_file", "source_hash"):
            continue
        base_val = merged.get(key)
        if isinstance(val, list) and isinstance(base_val, list):
            seen = {json.dumps(x, ensure_ascii=False, sort_keys=True) for x in base_val}
            for item in val:
                rep = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if rep not in seen:
                    base_val.append(item)
                    seen.add(rep)
            merged[key] = base_val
        elif isinstance(val, dict) and isinstance(base_val, dict):
            merged[key] = {**base_val, **{k: v for k, v in val.items() if v is not None}}
        elif val not in (None, "", [], {}):
            if base_val in (None, "", [], {}):
                merged[key] = val
    return merged


# ---------------------------------------------------------------------------
# File-level entry point
# ---------------------------------------------------------------------------

async def ingest_file(
    txt_path: Path,
    case_id: str | None = None,
    client: anthropic.AsyncAnthropic | None = None,
    model: str | None = None,
) -> ClinicalCase:
    """
    Ingest a single .txt case file.
    Returns a validated ClinicalCase object (not yet de-identified).
    """
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    raw_text = txt_path.read_text(encoding="utf-8", errors="replace")
    case_id = case_id or _derive_case_id(txt_path)
    client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    logger.info(f"[{case_id}] Ingesting {txt_path.name} ({len(raw_text)} chars)")
    case = await extract_from_text(raw_text, case_id, str(txt_path), client, model)
    logger.success(f"[{case_id}] Extraction complete, confidence={case.extraction_confidence:.2f}")
    return case


def _derive_case_id(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)[:32]
    suffix = uuid.uuid4().hex[:6].upper()
    return f"{stem}_{suffix}"


async def ingest_batch(
    txt_paths: list[Path],
    client: anthropic.AsyncAnthropic | None = None,
    model: str | None = None,
    max_concurrent: int | None = None,
) -> list[ClinicalCase]:
    """Ingest a batch of txt files concurrently (respecting max_concurrent)."""
    import asyncio
    from asyncio import Semaphore

    client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    sem = Semaphore(max_concurrent or settings.max_concurrent)

    async def _one(p: Path) -> ClinicalCase | None:
        async with sem:
            try:
                return await ingest_file(p, client=client, model=model)
            except Exception as e:
                logger.error(f"Failed {p.name}: {e}")
                return None

    results = await asyncio.gather(*[_one(p) for p in txt_paths])
    return [r for r in results if r is not None]
