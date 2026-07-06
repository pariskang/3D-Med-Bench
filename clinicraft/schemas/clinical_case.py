"""Core clinical case schema — produced by Stage 1 ingestion."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Sex(str, Enum):
    M = "M"
    F = "F"
    O = "O"


class TerminologyCode(BaseModel):
    system: Literal["ICD-10", "ICD-11", "SNOMED-CT", "LOINC", "HPO", "RxNorm", "ATC"]
    code: str
    display: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Symptom(BaseModel):
    description: str
    onset: str | None = None
    duration: str | None = None
    location: str | None = None
    character: str | None = None  # burning, crushing, stabbing …
    aggravating: list[str] = []
    relieving: list[str] = []
    radiation: str | None = None
    severity_nrs: int | None = Field(default=None, ge=0, le=10)
    timing: str | None = None   # constant, intermittent, positional
    associated: list[str] = []
    codes: list[TerminologyCode] = []


class PhysicalFinding(BaseModel):
    system: str           # cardiovascular, respiratory, neuro …
    region: str
    finding: str
    normal: bool
    description: str
    codes: list[TerminologyCode] = []


class LabResult(BaseModel):
    test_name: str
    value: str
    unit: str | None = None
    reference_range: str | None = None
    abnormal: bool | None = None
    timestamp: str | None = None
    loinc_code: str | None = None
    loinc_display: str | None = None


class ImagingResult(BaseModel):
    modality: str           # CT, MRI, XR, US, PET, Echo …
    region: str
    findings: str
    impression: str
    timestamp: str | None = None
    dicom_ref: str | None = None


class Medication(BaseModel):
    name: str
    generic_name: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    indication: str | None = None
    rxnorm_code: str | None = None
    atc_code: str | None = None


class Diagnosis(BaseModel):
    name: str
    icd10_code: str | None = None
    icd11_code: str | None = None
    snomed_code: str | None = None
    hpo_codes: list[str] = []        # for rare/phenotypic diagnoses
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    type: Literal["definitive", "probable", "possible", "rule_out"] = "definitive"
    is_primary: bool = False


class TimelineEvent(BaseModel):
    relative_time: str    # e.g. "入院第3天", "发病后2周"
    event_type: Literal[
        "symptom_onset", "admission", "exam_finding", "lab_result",
        "imaging", "treatment", "complication", "discharge", "other"
    ]
    description: str
    clinical_significance: str | None = None


class ClinicalCase(BaseModel):
    """Structured clinical case extracted from raw text. Stage 1 output."""

    case_id: str
    source_file: str
    source_hash: str = ""            # xxhash of original text

    # Demographics (will be partially removed in Stage 2)
    age: int | None = None
    age_band: str | None = None      # set by de-id (e.g. "55-59岁"); age nulled
    sex: Sex | None = None
    ethnicity: str | None = None

    # Presentation
    chief_complaint: str = ""
    hpi: str = ""                    # History of Present Illness (现病史)
    pmh: list[str] = []             # Past Medical History (既往史)
    family_history: list[str] = []
    social_history: str | None = None
    allergies: list[str] = []
    medications: list[Medication] = []

    # Review of Systems
    ros: dict[str, list[str]] = {}   # system → [positive/negative findings]

    # Physical Examination
    vitals: dict[str, Any] = {}      # HR, BP, RR, SpO2, T, GCS …
    physical_exam: list[PhysicalFinding] = []

    # Investigations
    labs: list[LabResult] = []
    imaging: list[ImagingResult] = []
    other_investigations: list[dict] = []

    # Diagnoses
    diagnoses: list[Diagnosis] = []

    # Timeline
    timeline: list[TimelineEvent] = []

    # Outcome
    treatment_summary: str = ""
    outcome: str | None = None

    # Specialty / provenance (removed/generalized after de-id)
    specialty: str | None = None
    raw_hospital: str | None = None   # cleared in Stage 2
    raw_dates: list[str] = []         # cleared in Stage 2

    # Pipeline metadata
    extraction_model: str = "claude-opus-4-8"
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_ts: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = "3.0"

    @field_validator("vitals", mode="before")
    @classmethod
    def normalise_vitals(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, dict):
            normalised: dict[str, Any] = {}
            aliases = {
                "心率": "HR", "心率(次/分)": "HR",
                "血压": "BP", "收缩压": "SBP", "舒张压": "DBP",
                "呼吸": "RR", "呼吸频率": "RR",
                "血氧": "SpO2", "氧饱和度": "SpO2",
                "体温": "T", "脉搏": "pulse",
                "格拉斯哥": "GCS",
            }
            for k, val in v.items():
                key = aliases.get(k, k)
                normalised[key] = val
            return normalised
        return v
