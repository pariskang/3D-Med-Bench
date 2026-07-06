"""Unit tests for all Pydantic schemas."""

import pytest
from clinicraft.schemas.clinical_case import ClinicalCase, Diagnosis, Sex, LabResult
from clinicraft.schemas.ground_truth import GroundTruthGraph, DxEntry, LRPair
from clinicraft.schemas.interaction import Action, ActionType, Observation, PerceptionMode
from clinicraft.schemas.rubric import Rubric, RubricRequirement, CompletenessCheck
from clinicraft.schemas.case_pack import CasePack, WorldConfig, PatientConfig


def test_clinical_case_basic():
    case = ClinicalCase(
        case_id="TEST-001",
        source_file="test.txt",
        age=45,
        sex=Sex.M,
        chief_complaint="胸痛1小时",
        diagnoses=[Diagnosis(name="急性ST段抬高型心肌梗死", icd10_code="I21.0", is_primary=True)],
    )
    assert case.case_id == "TEST-001"
    assert case.sex == Sex.M
    assert len(case.diagnoses) == 1
    assert case.diagnoses[0].icd10_code == "I21.0"


def test_vitals_normalisation():
    """Chinese key names should be normalised to EN keys."""
    case = ClinicalCase(
        case_id="TEST-002",
        source_file="test.txt",
        vitals={"心率": 100, "血压": "130/80", "呼吸": 22},
    )
    assert "HR" in case.vitals
    assert case.vitals["HR"] == 100
    assert "RR" in case.vitals


def test_gtg_schema():
    gtg = GroundTruthGraph(
        case_id="TEST-001",
        problem_representation="急性单系统胸痛，自主神经症状，血流动力学不稳定",
        final_dx="STEMI",
        final_dx_icd10="I21.0",
        differential=[
            DxEntry(dx="STEMI", p_prior=0.55, must_not_miss=True),
            DxEntry(dx="主动脉夹层", p_prior=0.15, must_not_miss=True),
            DxEntry(dx="急性心包炎", p_prior=0.10),
        ],
        must_not_miss=["主动脉夹层"],
        red_flags=["血压不对称", "撕裂性胸痛"],
        expert_reasoning_trace=["Step1", "Step2"],
        atomic_facts=["胸痛1小时", "大汗", "血压92/60"],
    )
    assert gtg.final_dx == "STEMI"
    assert len(gtg.differential) == 3
    assert gtg.differential[0].must_not_miss is True


def test_action_constructors():
    a = Action.ask("您的胸痛放射到哪里？")
    assert a.action == ActionType.ASK
    assert "utterance" in a.params

    a2 = Action.auscultate("cardiac_apex")
    assert a2.action == ActionType.AUSCULTATE

    from clinicraft.schemas.interaction import DifferentialEntry
    a3 = Action.submit_differential([
        DifferentialEntry(dx="STEMI", p=0.6),
        DifferentialEntry(dx="NSTEMI", p=0.2),
    ])
    assert a3.action == ActionType.SUBMIT_DIFFERENTIAL
    assert len(a3.params["ranked"]) == 2


def test_rubric_schema():
    rubric = Rubric(
        case_id="TEST-001",
        completeness_check=CompletenessCheck(must_take_vitals=True),
        requirements=[
            RubricRequirement(id="r001", cat="C1", description="最终诊断命中", weight=3.0),
            RubricRequirement(id="r002", cat="C5", description="识别主动脉夹层红旗",
                              weight=3.0, veto_if_fail=True),
        ],
    )
    assert len(rubric.requirements) == 2
    assert rubric.hard_veto_ids() == ["r002"]


def test_case_pack_save_load(tmp_path):
    pack = CasePack(
        case_id="TEST-PACK-001",
        world_config=WorldConfig(seed=42, patient=PatientConfig(age=45, sex="M")),
    )
    pack.save(tmp_path / "TEST-PACK-001")
    loaded = CasePack.load(tmp_path / "TEST-PACK-001")
    assert loaded.case_id == "TEST-PACK-001"
    assert loaded.world_config.seed == 42


def test_observation_schema():
    from clinicraft.schemas.interaction import Channel, StructuredState, Vitals, Budget
    obs = Observation(
        turn=3,
        case_id="TEST-001",
        perception_mode=PerceptionMode.STRUCTURED_ONLY,
        channels=Channel(
            dialogue="我胸口很痛",
            structured_state=StructuredState(
                vitals=Vitals(HR=118, BP="92/60", SpO2=89.0),
                visible_signs=["diaphoresis", "pallor"],
            ),
        ),
        available_actions=["ask", "auscultate", "order_test"],
        budget=Budget(tokens_used=1000, tests_ordered=1),
    )
    assert obs.turn == 3
    assert obs.channels.structured_state.vitals.HR == 118
    assert "diaphoresis" in obs.channels.structured_state.visible_signs
