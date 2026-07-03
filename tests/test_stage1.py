"""Tests for Stage 1 ingestion (without live LLM calls)."""

import pytest
from clinicraft.pipeline.stage1_ingest import _chunk_text, _merge_extractions, _derive_case_id
from pathlib import Path


def test_chunk_text_short():
    text = "这是一段短文本"
    chunks = _chunk_text(text, max_chars=1000)
    assert chunks == [text]


def test_chunk_text_long():
    # Simulate a text with section headers
    text = (
        "主诉：胸痛1小时\n" * 100 +
        "现病史：患者1小时前突发胸痛\n" * 100 +
        "既往史：高血压10年\n" * 100 +
        "查体：BP 92/60 HR 118\n" * 100
    )
    chunks = _chunk_text(text, max_chars=500)
    assert len(chunks) > 1
    # All content should be preserved (approximately)
    combined = "".join(chunks)
    assert "胸痛" in combined


def test_merge_extractions_list_dedup():
    base = {"labs": [{"test_name": "血常规", "value": "WBC 12"}]}
    extra = {"labs": [
        {"test_name": "血常规", "value": "WBC 12"},  # duplicate
        {"test_name": "肝功能", "value": "ALT 85"},   # new
    ]}
    merged = _merge_extractions(base, extra)
    assert len(merged["labs"]) == 2  # deduped


def test_merge_extractions_scalar():
    base = {"chief_complaint": "", "age": None}
    extra = {"chief_complaint": "胸痛1小时", "age": 45}
    merged = _merge_extractions(base, extra)
    assert merged["chief_complaint"] == "胸痛1小时"
    assert merged["age"] == 45


def test_derive_case_id():
    path = Path("/data/cases/patient_001_discharge.txt")
    cid = _derive_case_id(path)
    assert cid.startswith("patient_001_discharge")
    assert "_" in cid  # has suffix


def test_derive_case_id_special_chars():
    path = Path("/data/cases/病例-2024-001.txt")
    cid = _derive_case_id(path)
    # Should not contain chars that break directory names
    import re
    assert re.match(r"^[A-Za-z0-9_-]+$", cid)
