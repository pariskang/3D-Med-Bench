"""CasePack — the fully packaged playable case (Stage 6 output)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class PhysioConfig(BaseModel):
    engine: Literal["pulse", "scripted", "none"] = "scripted"
    scenario_xml: str | None = None     # path to Pulse scenario XML
    dynamic_coverage: float = 0.0       # fraction of findings from live engine
    initial_state: dict[str, Any] = {}


class PatientConfig(BaseModel):
    age: int | None = None
    sex: str | None = None
    persona: Literal[
        "stoic", "anxious", "dismissive", "cooperative", "confused"
    ] = "cooperative"
    health_literacy: Literal["low", "mid", "high"] = "mid"
    disclosure: Literal[
        "full", "withholds_travel_history", "withholds_medication",
        "downplays_symptoms", "provides_misleading_info"
    ] = "full"
    temperament: str = ""
    npc_role: str = "patient"          # patient | nurse | family


class WorldConfig(BaseModel):
    seed: int = 42
    physio: PhysioConfig = Field(default_factory=PhysioConfig)
    patient: PatientConfig = Field(default_factory=PatientConfig)
    available_tests: list[str] = []    # tests the environment will answer
    available_imaging: list[str] = []
    max_turns: int = 40
    time_pressure: bool = False        # simulated real-time urgency
    perception_mode: str = "frame_stream"


class Strata(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] = "hard"
    rarity: Literal["common", "uncommon", "rare", "ultra_rare"] = "uncommon"
    error_prone: bool = False
    specialty: str = ""
    perception_tier: Literal["T1", "T2", "T3"] = "T1"
    dynamic_coverage: float = 0.0


class Source(BaseModel):
    provenance: str = "private_hospital"
    deid_standard: str = "GB/T42460-2023"
    contamination_free: bool = True
    ethics_approval: str = ""
    data_class_level: str = ""


class CasePack(BaseModel):
    """
    Complete self-contained playable case.
    Stored as a directory; this model represents the merged metadata.
    """
    case_id: str
    source: Source = Field(default_factory=Source)
    strata: Strata = Field(default_factory=Strata)
    world_config: WorldConfig = Field(default_factory=WorldConfig)

    # File references (relative to case directory)
    presentation_md: str = "presentation.md"
    ground_truth_ref: str = "ground_truth_graph.json"
    rubric_ref: str = "tests/rubric.json"

    schema_version: str = "3.0"

    @classmethod
    def load(cls, case_dir: Path) -> "CasePack":
        import json
        cfg = case_dir / "world_config.yaml"
        if not cfg.exists():
            raise FileNotFoundError(f"No world_config.yaml in {case_dir}")
        import yaml  # type: ignore[import-untyped]
        with open(cfg) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def save(self, case_dir: Path) -> None:
        import yaml  # type: ignore[import-untyped]
        case_dir.mkdir(parents=True, exist_ok=True)
        with open(case_dir / "world_config.yaml", "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)
