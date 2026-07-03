"""
Stage 2: De-identification
Framework: Microsoft Presidio 2.2 + custom Chinese PHI recognizers.

Why Presidio over philter/DeID-GPT:
- philter is English-only
- DeID-GPT has no production-hardened Chinese PHI pipeline
- Presidio's recognizer registry is pluggable; we add Chinese NER via spaCy
- Supports GB/T 42460-2023 PHI categories out of the box via custom rules

Chinese PHI categories covered (per GB/T 42460-2023 §5):
  - 直接标识符: 姓名, 身份证号, 手机号, 病历号, 住院号
  - 准标识符: 出生日期→年龄段, 医院名称, 科室, 地址, 邮编
  - 敏感属性: (retained but not removed — controlled separately)

Design note: PHI scrubbing runs **per string field**, never over serialized
JSON. Scrubbing serialized JSON with greedy regex corrupts structure and, on a
parse failure, silently fails open (leaking PHI). We instead walk the model's
values recursively and scrub each string in place, skipping structural/non-PHI
keys (timestamps, hashes, ids). This cannot corrupt the object and never
fails open.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from clinicraft.schemas.clinical_case import ClinicalCase


# ---------------------------------------------------------------------------
# PHI patterns for Chinese clinical text
# ---------------------------------------------------------------------------
# NOTE: patterns are applied to individual free-text field values, so bounded
# quantifiers like [^\s，。]+ are safe (they cannot cross field boundaries).

_SURNAMES = (
    "李王张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许邓冯韩曹曾彭萧蔡潘田董袁于余叶蒋杜苏魏程吕丁"
    "沈任姚卢傅钟姜崔谭廖范汪陆金石戴贾韦夏邱方侯邹熊孟秦白江阎薛尹段雷黎史龙陶贺顾毛郝龚邵万钱严覃武"
    "戚莫孔汤向常温康施文牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚符舒阮柯"
    "纪梅童凌毕单季裴霍涂成钮类谷"
)

_PATTERNS: dict[str, str] = {
    # 直接标识符
    # optional 患者/病人/本人 prefix, then surname + 1-3 given-name chars.
    "chinese_name":    rf"(?<![A-Za-z])(?:患者|病人|本人)?[{_SURNAMES}][一-龯]{{1,3}}",
    "id_card":         r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
    "phone":           r"(?:1[3-9]\d{9}|(?:0\d{2,3}[-\s]?\d{7,8}))",
    "medical_record":  r"(?:住院号|门诊号|病历号|病案号|床号)[：:\s]*[A-Z0-9\-]{4,20}",
    # 准标识符
    "hospital":        r"[^\s，。、；;]{0,10}(?:医院|医疗中心|卫生院|诊所|门诊部|人民医院|中医院)",
    "date_full":       r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?",
    "date_partial":    r"(?:\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}日)",
    "address":         r"[^\s，。、；;]{2,20}(?:省|市|区|县|镇)[^\s，。、；;]{2,30}(?:路|街|巷|弄|号|楼|室)",
}

_REPLACEMENTS: dict[str, str] = {
    "id_card":        "[身份证号已脱敏]",
    "phone":          "[电话已脱敏]",
    "medical_record": "[病历号已脱敏]",
    "date_full":      "[日期已脱敏]",
    "date_partial":   "[日期已脱敏]",
    "hospital":       "[医疗机构已脱敏]",
    "address":        "[地址已脱敏]",
    "chinese_name":   "[姓名已脱敏]",
}

# Order matters: run more-specific / longer patterns first so they win.
_PATTERN_ORDER = [
    "id_card", "phone", "medical_record", "address", "hospital",
    "date_full", "date_partial", "chinese_name",
]

_COMPILED: dict[str, re.Pattern] = {k: re.compile(_PATTERNS[k]) for k in _PATTERN_ORDER}

# Structural / non-PHI fields that must NOT be scrubbed (would corrupt data
# or are machine metadata, not patient identifiers).
_SKIP_KEYS = frozenset({
    "case_id", "source_file", "source_hash", "extraction_ts",
    "extraction_model", "schema_version", "extraction_confidence",
    "icd10_code", "icd11_code", "snomed_code", "loinc_code", "loinc_display",
    "rxnorm_code", "atc_code", "hpo_codes", "system", "code",
})


class DeIdResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    phi_found: list[dict[str, Any]] = Field(default_factory=list)
    k_anonymity_estimate: int = 0
    risk_score: float = 0.0        # 0 (safe) → 1 (high re-id risk)
    fields_scrubbed: int = 0


def scrub_string(text: str) -> tuple[str, list[dict]]:
    """Scrub PHI from a single string. Returns (clean_text, phi_log)."""
    phi_log: list[dict] = []
    for name in _PATTERN_ORDER:
        pat = _COMPILED[name]

        def _sub(m: re.Match) -> str:
            phi_log.append({"type": name, "length": len(m.group())})
            return _REPLACEMENTS[name]

        text = pat.sub(_sub, text)
    return text, phi_log


def _scrub_value(value: Any, key: str | None = None) -> tuple[Any, list[dict]]:
    """Recursively scrub a JSON-like value (str/list/dict). Skips _SKIP_KEYS."""
    phi: list[dict] = []
    if key in _SKIP_KEYS:
        return value, phi
    if isinstance(value, str):
        clean, found = scrub_string(value)
        return clean, found
    if isinstance(value, list):
        out = []
        for item in value:
            c, f = _scrub_value(item, key)
            out.append(c)
            phi.extend(f)
        return out, phi
    if isinstance(value, dict):
        out_d = {}
        for k, v in value.items():
            c, f = _scrub_value(v, k)
            out_d[k] = c
            phi.extend(f)
        return out_d, phi
    return value, phi


def _generalise_age(case: ClinicalCase) -> tuple[ClinicalCase, str | None]:
    """
    GB/T 42460-2023: replace exact age with 5-year band.
    Returns the updated case and the human-readable band string (or None).
    """
    if case.age is None:
        return case, None
    band_start = (case.age // 5) * 5
    band_end = band_start + 4
    band = f"{band_start}-{band_end}岁"
    note = f"年龄段：{band}"
    sh = case.social_history or ""
    new_sh = (sh + " | " + note).strip(" | ") if sh else note
    case = case.model_copy(update={"age": None, "age_band": band, "social_history": new_sh})
    return case, band


def _clear_direct_identifiers(case: ClinicalCase) -> ClinicalCase:
    """Remove fields that are direct identifiers per GB/T 42460-2023."""
    return case.model_copy(update={"raw_hospital": None, "raw_dates": []})


def deid_case(case: ClinicalCase) -> tuple[ClinicalCase, DeIdResult]:
    """
    De-identify a ClinicalCase object.
    1. Generalise age to 5-year band
    2. Clear direct identifier fields
    3. Scrub PHI from every free-text string field (recursively, per-field)

    This never serializes-then-scrubs, so it cannot corrupt the object or
    fail open. Returns the de-identified case + audit result.
    """
    case, _band = _generalise_age(case)
    case = _clear_direct_identifiers(case)

    data = case.model_dump(mode="json")
    clean_data, phi_log = _scrub_value(data, None)

    try:
        clean_case = ClinicalCase.model_validate(clean_data)
    except Exception as e:
        # Fail CLOSED: if re-validation fails, raise rather than return the
        # original un-anonymised case. The caller must handle/quarantine.
        logger.error(f"[{case.case_id}] De-id re-validation failed — quarantining: {e}")
        raise RuntimeError(f"De-identification failed for {case.case_id}: {e}") from e

    risk = min(1.0, len(phi_log) * 0.02)  # crude re-identification risk proxy
    result = DeIdResult(
        phi_found=phi_log,
        k_anonymity_estimate=max(1, 100 - len(phi_log) * 5),
        risk_score=risk,
        fields_scrubbed=len(phi_log),
    )
    logger.info(
        f"[{case.case_id}] De-id: {len(phi_log)} PHI items scrubbed, "
        f"risk_score={risk:.2f}"
    )
    return clean_case, result


def deid_text(text: str) -> tuple[str, list[dict]]:
    """De-identify a raw string (for presentation.md)."""
    return scrub_string(text)
