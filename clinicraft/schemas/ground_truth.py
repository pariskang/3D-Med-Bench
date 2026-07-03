"""Ground Truth Graph (GTG) schema — produced by Stage 3, expert-validated."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DxEntry(BaseModel):
    """One differential diagnosis entry with prior probability."""
    dx: str
    icd10: str | None = None
    p_prior: float = Field(ge=0.0, le=1.0)       # prior probability
    supporting: list[str] = []                     # supporting features
    against: list[str] = []                        # features arguing against
    must_not_miss: bool = False


class LRPair(BaseModel):
    """Likelihood ratio for a clinical finding re a target diagnosis."""
    dx: str
    lr_pos: float | None = None    # LR+ (finding present → dx probability ↑)
    lr_neg: float | None = None    # LR− (finding absent → dx probability ↓)
    source: str = "JAMA-RCE"       # evidence source


class WorkupStep(BaseModel):
    """One ideal investigation step with cost and LR information."""
    test: str
    loinc: str | None = None
    rationale: str
    pre_test_p: float | None = Field(default=None, ge=0.0, le=1.0)
    lr_pairs: list[LRPair] = []
    cost_usd: float | None = None
    choosing_wisely_concern: bool = False
    timing: Literal["immediate", "urgent", "elective"] = "urgent"


class VisibleSign(BaseModel):
    """A body sign renderable in the 3D avatar (maps to Sign Rendering Library)."""
    sign_id: str               # key in signs.yaml
    description: str
    region: str
    severity: Literal["mild", "moderate", "severe"] = "moderate"
    render_tier: Literal["T1", "T2", "T3"] = "T1"
    lr_pairs: list[LRPair] = []


class ManagementStep(BaseModel):
    action: str
    rationale: str
    guideline_ref: str | None = None
    timing: Literal["immediate", "urgent", "elective"] = "urgent"
    contraindication_if_missed: str | None = None


class GroundTruthGraph(BaseModel):
    """
    Expert-validated ground truth for one case.
    Built by Stage 3, validated by ≥2 specialist physicians.
    Used by judge, oracle agent, and scoring rubric.
    """

    case_id: str

    # §2.1 Problem representation (one-sentence semantic summary)
    problem_representation: str

    # Atomic facts extracted from case (used by hallucination guard)
    atomic_facts: list[str] = []

    # Final diagnosis
    final_dx: str
    final_dx_icd10: str | None = None
    hpo_codes: list[str] = []        # HP:XXXXXXX codes for phenotypic diagnosis

    # Differential (ranked by prior probability)
    differential: list[DxEntry] = []

    # Must-not-miss diagnoses (C5 safety gate)
    must_not_miss: list[str] = []

    # Red flags that should trigger escalation
    red_flags: list[str] = []

    # Ideal workup sequence
    ideal_workup: list[WorkupStep] = []

    # Visible signs (for 3D rendering / C3 assessment)
    visible_signs: list[VisibleSign] = []

    # Management
    management_plan: list[ManagementStep] = []
    safety_net_items: list[str] = []      # safety-netting instructions

    # §2.1 Diagnostic threshold model
    test_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    treatment_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    # §2.4 Expert reasoning trace (承 MedCaseReasoning)
    expert_reasoning_trace: list[str] = []

    # Stratification labels (Stage 7)
    difficulty: Literal["easy", "medium", "hard"] = "hard"
    rarity: Literal["common", "uncommon", "rare", "ultra_rare"] = "uncommon"
    error_prone: bool = False
    specialty: str = ""
    perception_tier: Literal["T1", "T2", "T3"] = "T1"
    dynamic_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    # Steering traps (C5): incorrect suggestions that should NOT sway the model
    steering_traps: list[dict[str, Any]] = []

    # Validation metadata
    expert_1: str = ""
    expert_2: str = ""
    arbitrator: str | None = None
    validated: bool = False
    schema_version: str = "3.0"
