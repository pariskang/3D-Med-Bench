"""
Patient Host (Layer 2).
Plays the patient NPC during an encounter. Key responsibilities:
1. Respond to doctor questions within atomic_facts (hallucination guard)
2. Maintain persona/temperament consistency across turns
3. Selectively disclose based on disclosure setting
4. Trigger scheduled events (symptom worsening, spontaneous information)
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from clinicraft.config import settings
from clinicraft.patient.hallucination_guard import HallucinationGuard
from clinicraft.patient.memory import MemoryStream
from clinicraft.schemas.case_pack import PatientConfig
from clinicraft.schemas.ground_truth import GroundTruthGraph


_SYSTEM_PATIENT = """你正在扮演一名真实患者。
你的回答必须严格来自你的"已知事实"范围（atomic_facts），不能编造、推断或超出。
如果医生问到你不知道的内容，诚实地说"我不太清楚"或"没注意到"。
根据你的性格（{persona}）和健康素养（{health_literacy}）来表达。
不要主动透露诊断名称或医学术语，除非你的角色设定包含医学背景。
当被直接询问某症状是否存在时，严格按atomic_facts回答yes/no/不确定。
你的语言风格：{language_style}"""


class PatientHost:
    """LLM-driven patient NPC with atomic_facts grounding."""

    def __init__(
        self,
        gtg: GroundTruthGraph,
        config: PatientConfig,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._gtg = gtg
        self._config = config
        self._client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._memory = MemoryStream()
        self._guard = HallucinationGuard(gtg.atomic_facts)
        self._turn = 0
        self.tokens = 0

    def _build_system(self) -> str:
        style = {
            "low": "用简单直白的语言，可能描述不清楚症状",
            "mid": "普通语言，基本能描述症状",
            "high": "相对准确，可能使用一些医学词汇",
        }.get(self._config.health_literacy, "普通语言")
        return _SYSTEM_PATIENT.format(
            persona=self._config.persona,
            health_literacy=self._config.health_literacy,
            language_style=style,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def respond(self, doctor_utterance: str) -> str:
        """Generate patient response to a doctor utterance."""
        self._turn += 1
        self._memory.add_doctor(doctor_utterance, self._turn)

        # Build context: atomic_facts + recent conversation
        facts_block = "\n".join(f"- {f}" for f in self._gtg.atomic_facts)
        disclosure_note = self._get_disclosure_note()

        messages = [
            {
                "role": "user",
                "content": (
                    f"[已知事实范围]\n{facts_block}\n\n"
                    f"[透露设定] {disclosure_note}\n\n"
                    f"[近期对话]\n{self._memory.recent_context(n=6)}\n\n"
                    f"[医生说] {doctor_utterance}\n\n"
                    "请以患者身份作答（1-3句，口语化，不超出已知事实范围）："
                ),
            }
        ]

        resp = await self._client.messages.create(
            model=settings.llm_model,
            max_tokens=512,
            system=self._build_system(),
            messages=messages,
        )
        if resp.usage:
            self.tokens += resp.usage.input_tokens + resp.usage.output_tokens
        patient_text = resp.content[0].text.strip()

        # Hallucination check
        if self._guard.is_hallucination(doctor_utterance, patient_text):
            logger.warning(f"[{self._gtg.case_id}] Turn {self._turn}: hallucination detected, grounding")
            patient_text = self._guard.ground_response(patient_text)

        self._memory.add_patient(patient_text, self._turn)
        return patient_text

    def _get_disclosure_note(self) -> str:
        disc = self._config.disclosure
        if disc == "full":
            return "如实回答所有问题"
        elif disc == "withholds_travel_history":
            return "隐瞒旅行史，除非被直接追问3次"
        elif disc == "withholds_medication":
            return "隐瞒正在服用的某种药物，除非直接问到药物过敏或用药史"
        elif disc == "downplays_symptoms":
            return "倾向于淡化症状严重程度，说'没那么严重'"
        elif disc == "provides_misleading_info":
            return "初次陈述时提供一个误导性信息（与atomic_facts不符的细节）"
        return ""

    def get_memory(self) -> MemoryStream:
        return self._memory
