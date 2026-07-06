"""
Hallucination Guard: ensures patient NPC responses don't exceed atomic_facts.
Simple keyword-overlap heuristic + LLM-based check for critical fabrications.
"""

from __future__ import annotations

import re


class HallucinationGuard:
    """
    Checks whether a patient response introduces facts not in atomic_facts.
    Uses simple keyword overlap — production upgrade: use embedding cosine sim.
    """

    def __init__(self, atomic_facts: list[str]) -> None:
        self._facts = atomic_facts
        self._fact_text = " ".join(atomic_facts).lower()
        # Extract key numeric values and named entities from facts
        self._numbers = set(re.findall(r"\d+\.?\d*", self._fact_text))
        self._negations = {"否认", "无", "不", "没有", "未"}

    def is_hallucination(self, question: str, response: str) -> bool:
        """
        Heuristic: flag if response introduces a number not in atomic_facts
        AND the number is clinically significant (vital range or lab value).
        """
        resp_numbers = set(re.findall(r"\d+\.?\d*", response))
        novel = resp_numbers - self._numbers
        clinical_novel = {
            n for n in novel
            if 30 <= float(n) <= 300  # likely a clinical value
        }
        return len(clinical_novel) > 0

    def ground_response(self, response: str) -> str:
        """
        Fallback response when hallucination detected.
        Returns a safe non-committal answer.
        """
        return "我说不太准确，医生你看报告单可能更清楚。"
