"""
§5 Typed observation-action loop schemas.

Every turn: environment sends Observation → model returns Action.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class PerceptionMode(str, Enum):
    FRAME_STREAM = "frame_stream"         # model must perceive signs from video
    STRUCTURED_ONLY = "structured_only"   # signs given as text (baseline)
    DUAL = "dual"                         # both (ablation: gain from vision)


# ---------------------------------------------------------------------------
# Observation (env → model)
# ---------------------------------------------------------------------------

class VisionChannel(BaseModel):
    frames: list[str] = []       # image refs or base64 data URIs
    fps: float = 4.0
    view: str = "patient_front"  # patient_front | close_up_face | full_body


class AudioChannel(BaseModel):
    clip_ref: str | None = None
    may_contain: list[str] = []  # speech | dyspnea | wheeze | stridor | cough


class Vitals(BaseModel):
    HR: int | None = None
    BP: str | None = None        # "120/80"
    RR: int | None = None
    SpO2: float | None = None
    T: float | None = None
    GCS: int | None = None
    extra: dict[str, Any] = {}


class StructuredState(BaseModel):
    vitals: Vitals = Field(default_factory=Vitals)
    visible_signs: list[str] = []   # EMPTY in frame_stream mode — model must observe
    patient_posture: str | None = None
    patient_distress_level: Literal["none", "mild", "moderate", "severe"] | None = None


class ActionResult(BaseModel):
    """Result of the previous model action, included in next observation."""
    action: str
    site: str | None = None
    finding: str | None = None
    audio_ref: str | None = None
    image_ref: str | None = None
    lr_pairs: list[dict[str, float]] = []   # {dx: lr_value}
    error: str | None = None


class Budget(BaseModel):
    tokens_used: int = 0
    tests_ordered: int = 0
    sim_cost_cny: float = 0.0   # accumulated simulated cost in CNY


class Channel(BaseModel):
    vision: VisionChannel | None = None
    audio: AudioChannel | None = None
    dialogue: str | None = None
    structured_state: StructuredState = Field(default_factory=StructuredState)
    last_action_result: ActionResult | None = None


class Observation(BaseModel):
    """Complete environment observation delivered to the model each turn."""
    turn: int
    case_id: str
    perception_mode: PerceptionMode = PerceptionMode.FRAME_STREAM
    channels: Channel
    available_actions: list[str]
    clock: dict[str, Any] = {}        # sim_minutes_elapsed, wall_seconds
    budget: Budget = Field(default_factory=Budget)
    episode_done: bool = False


# ---------------------------------------------------------------------------
# Actions (model → env) — §5.3
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    # Communication
    ASK = "ask"
    COUNSEL = "counsel"
    SAFETY_NET = "safety_net"

    # Active perception (avatar executes, re-renders)
    OBSERVE_TASK = "observe_task"     # e.g. "walk_10m", "write_name"
    INSPECT = "inspect"               # visual inspection of region

    # Examination maneuvers (return finding + LR + optional audio/image)
    AUSCULTATE = "auscultate"
    PALPATE = "palpate"
    PERCUSS = "percuss"
    CHECK_REFLEX = "check_reflex"
    CHECK_PULSE = "check_pulse"
    CHECK_CAP_REFILL = "check_cap_refill"
    FUNDOSCOPY = "fundoscopy"
    OPHTHALMOSCOPY = "ophthalmoscopy"

    # TCM (望闻问切)
    TCM_INSPECT = "tcm_inspect"       # 望 (tongue, complexion)
    TCM_LISTEN = "tcm_listen"         # 闻 (voice, breath)
    TCM_PULSE = "tcm_pulse"           # 切 (pulse palpation)

    # Investigations
    ORDER_TEST = "order_test"
    ORDER_IMAGING = "order_imaging"
    ORDER_PROCEDURE = "order_procedure"

    # Treatment
    PRESCRIBE = "prescribe"
    REFER = "refer"
    ESCALATE = "escalate"

    # Cognitive / metacognitive (§2.1 assessment targets)
    SUBMIT_PROBLEM_REP = "submit_problem_rep"
    SUBMIT_DIFFERENTIAL = "submit_differential"
    CHOOSE_NEXT_STEP = "choose_next_step"
    SUBMIT_DIAGNOSIS = "submit_diagnosis"
    SUBMIT_PLAN = "submit_plan"
    EXPRESS_UNCERTAINTY = "express_uncertainty"
    REQUEST_SENIOR = "request_senior"


class DifferentialEntry(BaseModel):
    dx: str
    p: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class Action(BaseModel):
    """Typed action emitted by the doctor model each turn."""
    action: ActionType
    params: dict[str, Any] = {}

    # Convenience typed accessors built as class methods ----------------

    @classmethod
    def ask(cls, utterance: str, style: str = "open") -> "Action":
        return cls(action=ActionType.ASK, params={"utterance": utterance, "style": style})

    @classmethod
    def auscultate(cls, site: str) -> "Action":
        return cls(action=ActionType.AUSCULTATE, params={"site": site})

    @classmethod
    def palpate(cls, region: str, maneuver: str | None = None) -> "Action":
        p = {"region": region}
        if maneuver:
            p["maneuver"] = maneuver
        return cls(action=ActionType.PALPATE, params=p)

    @classmethod
    def order_test(cls, test: str, rationale: str = "") -> "Action":
        return cls(action=ActionType.ORDER_TEST, params={"test": test, "rationale": rationale})

    @classmethod
    def submit_differential(cls, ranked: list[DifferentialEntry]) -> "Action":
        return cls(
            action=ActionType.SUBMIT_DIFFERENTIAL,
            params={"ranked": [e.model_dump() for e in ranked]},
        )

    @classmethod
    def submit_diagnosis(cls, dx: str, confidence: float, rationale: str) -> "Action":
        return cls(
            action=ActionType.SUBMIT_DIAGNOSIS,
            params={"dx": dx, "confidence": confidence, "rationale": rationale},
        )

    @classmethod
    def express_uncertainty(cls, confidence: float, would_defer: bool) -> "Action":
        return cls(
            action=ActionType.EXPRESS_UNCERTAINTY,
            params={"confidence": confidence, "would_defer": would_defer},
        )
