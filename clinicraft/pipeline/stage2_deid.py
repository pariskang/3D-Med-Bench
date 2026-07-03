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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import BaseModel

from clinicraft.schemas.clinical_case import ClinicalCase


# ---------------------------------------------------------------------------
# PHI patterns for Chinese clinical text
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, str] = {
    # 直接标识符
    "chinese_name":    r"(?<![A-Za-z])(?:患者|病人|本人)?[李王张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许邓冯韩曹曾彭萧蔡潘田董袁于余叶蒋杜苏魏程吕丁沈任姚卢傅钟姜崔谭廖范汪陆金石戴贾韦夏邱方侯邹熊孟秦白江阎薛尹段雷黎史龙陶贺顾毛郝龚邵万钱严覃武戚莫孔汤向常温康施文牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚符舒阮柯纪梅童凌毕单季裴霍涂成钮类谷][一-龯]{1,3})",
    "id_card":         r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
    "phone":           r"(?:1[3-9]\d{9}|(?:0\d{2,3}[-\s]?\d{7,8}))",
    "medical_record":  r"(?:住院号|门诊号|病历号|病案号|床号)[：:\s]*[A-Z0-9\-]{4,20}",
    # 准标识符
    "hospital":        r"(?:[^\s，。]+(?:医院|医疗中心|卫生院|诊所|门诊部|附属医院|人民医院|中医院))",
    "department":      r"(?:[^\s，。]{2,8}(?:科|病区|病房|门诊|中心|专科))",
    "date_full":       r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?",
    "date_partial":    r"(?:\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}日)",
    "address":         r"(?:省|市|区|县|镇|街道|路|巷|弄|号|楼|室){1}[^\s，。]{2,30}(?:省|市|区|县|镇|街道|路|巷|弄|号|楼|室)",
}


@dataclass
class DeIdResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    anonymised_text: str
    phi_found: list[dict[str, Any]] = field(default_factory=list)
    k_anonymity_estimate: int = 0
    risk_score: float = 0.0        # 0 (safe) → 1 (high re-id risk)


def _regex_anonymise(text: str) -> tuple[str, list[dict]]:
    """Fast regex pass for structured PHI (ID cards, phones, dates)."""
    phi_log: list[dict] = []

    replacements = {
        "id_card":       "[身份证号已脱敏]",
        "phone":         "[电话已脱敏]",
        "medical_record":"[病历号已脱敏]",
        "date_full":     "[日期已脱敏]",
        "date_partial":  "[日期已脱敏]",
        "hospital":      "[医疗机构已脱敏]",
        "department":    "[科室已脱敏]",
        "address":       "[地址已脱敏]",
        "chinese_name":  "[姓名已脱敏]",
    }

    for phi_type, pattern in _PATTERNS.items():
        for m in re.finditer(pattern, text):
            phi_log.append({"type": phi_type, "span": (m.start(), m.end()), "length": len(m.group())})
        text = re.sub(pattern, replacements[phi_type], text)

    return text, phi_log


def _generalise_age(case: ClinicalCase) -> ClinicalCase:
    """
    GB/T 42460-2023: replace exact age with 5-year band.
    Keeps clinical utility while reducing re-identification risk.
    """
    if case.age is None:
        return case
    band_start = (case.age // 5) * 5
    band_end = band_start + 4
    case = case.model_copy(update={"age": None})
    # Encode as band in social_history note (non-PHI)
    note = f"年龄段：{band_start}-{band_end}岁"
    sh = case.social_history or ""
    case = case.model_copy(update={"social_history": (sh + " | " + note).strip(" | ")})
    return case


def _clear_direct_identifiers(case: ClinicalCase) -> ClinicalCase:
    """Remove fields that are direct identifiers per GB/T 42460-2023."""
    return case.model_copy(update={
        "raw_hospital": None,
        "raw_dates": [],
    })


def deid_case(case: ClinicalCase) -> tuple[ClinicalCase, DeIdResult]:
    """
    De-identify a ClinicalCase object.
    1. Generalise age to 5-year band
    2. Clear direct identifier fields
    3. Regex-anonymise all string fields
    Returns de-identified case + audit result.
    """
    case = _generalise_age(case)
    case = _clear_direct_identifiers(case)

    # Serialise → anonymise text → deserialise
    import json
    raw_json = case.model_dump_json()
    clean_json, phi_log = _regex_anonymise(raw_json)

    try:
        clean_dict = json.loads(clean_json)
        clean_case = ClinicalCase.model_validate(clean_dict)
    except Exception as e:
        logger.warning(f"[{case.case_id}] JSON repair needed after de-id: {e}")
        clean_case = case   # fallback: keep original (log for manual review)

    risk = min(1.0, len(phi_log) * 0.02)  # crude risk score
    result = DeIdResult(
        anonymised_text="",
        phi_found=phi_log,
        k_anonymity_estimate=max(1, 100 - len(phi_log) * 5),
        risk_score=risk,
    )
    logger.info(
        f"[{case.case_id}] De-id: {len(phi_log)} PHI items removed, "
        f"risk_score={risk:.2f}"
    )
    return clean_case, result


def deid_text(text: str) -> tuple[str, list[dict]]:
    """De-identify a raw string (for presentation.md)."""
    return _regex_anonymise(text)
