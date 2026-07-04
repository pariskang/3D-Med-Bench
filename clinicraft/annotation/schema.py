"""Annotation schema — one expert's double-blind review of one case's GTG draft."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    ACCEPT = "accept"      # draft field is clinically correct as-is
    EDIT = "edit"          # correct but needs the supplied corrected_value
    REJECT = "reject"      # wrong / unusable


class FieldJudgment(BaseModel):
    """An expert's verdict on one GTG field."""
    verdict: Verdict = Verdict.ACCEPT
    corrected_value: Any = None       # required iff verdict == EDIT
    comment: str = ""


class StrataJudgment(BaseModel):
    """The expert's *independent* stratification labels (for κ vs the other expert)."""
    difficulty: str = "hard"          # easy | medium | hard  (ordinal)
    rarity: str = "uncommon"          # common | uncommon | rare | ultra_rare (ordinal)
    error_prone: bool = False


# The GTG fields experts review (used by consensus + IRR).
REVIEWABLE_FIELDS = [
    "problem_representation", "final_dx", "differential", "must_not_miss",
    "red_flags", "ideal_workup", "visible_signs", "management_plan",
    "expert_reasoning_trace",
]

# Categorical variables on which inter-rater reliability (κ) is computed.
# name → ("binary" | "ordinal", ordered categories or None)
IRR_VARIABLES: dict[str, tuple[str, list | None]] = {
    "overall_valid":  ("binary", None),
    "final_dx_agree": ("binary", None),
    "error_prone":    ("binary", None),
    "difficulty":     ("ordinal", ["easy", "medium", "hard"]),
    "rarity":         ("ordinal", ["common", "uncommon", "rare", "ultra_rare"]),
}


class GTGAnnotation(BaseModel):
    """One expert's complete review of one case's GTG draft."""
    case_id: str
    annotator_id: str
    role: str = "specialist"           # specialist | resident | arbitrator
    specialty: str = ""
    blinded: bool = True               # reviewer did not see other reviews

    # Top-level clinical judgments (drive κ)
    overall_valid: bool = True         # is the GTG clinically sound overall?
    final_dx_agree: bool = True        # does drafted final_dx match expert's view?
    final_dx_corrected: str | None = None

    # Independent stratification labels
    strata: StrataJudgment = Field(default_factory=StrataJudgment)

    # Per-field verdicts
    field_judgments: dict[str, FieldJudgment] = Field(default_factory=dict)

    free_comment: str = ""
    submitted_ts: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = "3.0"

    def irr_labels(self) -> dict[str, Any]:
        """Extract the categorical labels used for inter-rater reliability."""
        return {
            "overall_valid": self.overall_valid,
            "final_dx_agree": self.final_dx_agree,
            "error_prone": self.strata.error_prone,
            "difficulty": self.strata.difficulty,
            "rarity": self.strata.rarity,
        }


class AnnotationTask(BaseModel):
    """A pre-filled review form handed to an expert (draft summary + blank slots)."""
    case_id: str
    specialty: str = ""
    draft_final_dx: str = ""
    draft_problem_representation: str = ""
    draft_differential: list[dict] = []
    draft_must_not_miss: list[str] = []
    draft_red_flags: list[str] = []
    draft_difficulty: str = "hard"
    draft_rarity: str = "uncommon"
    draft_error_prone: bool = False
    instructions: str = (
        "请对以下AI起草的病例真值图进行双盲评审。对每个字段给出 accept/edit/reject，"
        "并独立标注难度/罕见度/易错性。请勿参考其他评审者意见。"
    )
