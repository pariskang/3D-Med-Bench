"""
Regression tests for the audit findings (F1-F8, interaction #1-#13).
Each test pins a specific bug that was fixed, so it cannot silently return.
All run without an API key (oracle/nop agents; patient never invoked by oracle).
"""

import re
import asyncio
from pathlib import Path

import pytest

from clinicraft.schemas.clinical_case import ClinicalCase, Sex
from clinicraft.schemas.ground_truth import GroundTruthGraph


# --------------------------------------------------------------------------
# Stage 2 de-identification (F1: dataclass crash, F2: fail-open, name regex)
# --------------------------------------------------------------------------

def test_deid_regex_all_compile():
    """chinese_name previously had an unbalanced paren → crashed all de-id."""
    from clinicraft.pipeline.stage2_deid import _PATTERNS
    for name, pat in _PATTERNS.items():
        re.compile(pat)  # must not raise


def test_deid_result_constructs():
    """F1: DeIdResult was @dataclass on BaseModel → AttributeError on construct."""
    from clinicraft.pipeline.stage2_deid import DeIdResult
    r = DeIdResult(phi_found=[{"type": "phone"}], k_anonymity_estimate=50, risk_score=0.1)
    assert r.risk_score == 0.1
    assert r.fields_scrubbed == 0


def test_deid_scrubs_and_does_not_crash():
    case = ClinicalCase(
        case_id="D1", source_file="t.txt", age=58, sex=Sex.M,
        chief_complaint="胸痛",
        hpi="患者李明，手机13800138000，2024年3月5日就诊于北京协和医院。",
        raw_hospital="北京协和医院",
    )
    from clinicraft.pipeline.stage2_deid import deid_case
    clean, report = deid_case(case)
    # Age generalised to band, exact age nulled
    assert clean.age is None
    assert clean.age_band is not None and "岁" in clean.age_band
    # PHI scrubbed from free text
    assert "13800138000" not in clean.hpi
    assert "2024年3月5日" not in clean.hpi
    assert "李明" not in clean.hpi
    assert report.fields_scrubbed > 0


def test_deid_does_not_corrupt_structured_codes():
    """Per-field scrubbing must leave ICD/LOINC codes intact (skip keys)."""
    from clinicraft.schemas.clinical_case import Diagnosis
    case = ClinicalCase(
        case_id="D2", source_file="t.txt", age=40,
        diagnoses=[Diagnosis(name="STEMI", icd10_code="I21.0")],
    )
    from clinicraft.pipeline.stage2_deid import deid_case
    clean, _ = deid_case(case)
    assert clean.diagnoses[0].icd10_code == "I21.0"


def test_deid_fails_closed(monkeypatch):
    """F2: on re-validation failure, de-id must RAISE, not return original PHI."""
    from clinicraft.pipeline import stage2_deid
    case = ClinicalCase(case_id="D3", source_file="t.txt", age=30, chief_complaint="x")

    def _boom(data, key=None):
        return {"broken": object()}, []  # unvalidatable
    monkeypatch.setattr(stage2_deid, "_scrub_value", _boom)
    with pytest.raises(RuntimeError):
        stage2_deid.deid_case(case)


# --------------------------------------------------------------------------
# Interaction loop (#1 reset NameError, #2 fractional HR, #3 dup site)
# --------------------------------------------------------------------------

def _mini_case_pack(tmp_path, mode):
    from clinicraft.pipeline.stage6_pack import pack_case
    from clinicraft.schemas.case_pack import CasePack
    case = ClinicalCase(case_id="EP", source_file="t.txt", age=58, sex=Sex.M,
                        chief_complaint="胸痛", vitals={"HR": 118, "BP": "92/60"})
    gtg = GroundTruthGraph(
        case_id="EP", problem_representation="pr", final_dx="STEMI", specialty="cardiology",
        differential=[{"dx": "STEMI", "p_prior": 0.6, "must_not_miss": True}],
        must_not_miss=["夹层"], red_flags=["rf"],
        visible_signs=[{"sign_id": "pallor", "description": "苍白", "region": "face"}],
        expert_reasoning_trace=["s1", "s2"],
        ideal_workup=[{"test": "ECG", "rationale": "r"}],
        management_plan=[{"action": "PCI", "rationale": "r"}],
        safety_net_items=["sn1"],
    )
    physio = {"engine": "scripted", "dynamic_coverage": 0.0,
              "initial_state": {"HR": 118, "SBP": 92, "DBP": 60, "SpO2": 94}}
    embody = {"perception_tier": "T1", "patient_config": {"persona": "anxious"},
              "avatar_spec": {}, "render_params": []}
    case_dir = pack_case(case, gtg, physio, embody, out_root=tmp_path)
    return case_dir, gtg, CasePack.load(case_dir)


async def _run_oracle(case_dir, gtg, pack, mode):
    from clinicraft.environment.clinical_env import ClinicalEnvironment
    from clinicraft.patient.host import PatientHost
    from clinicraft.physio.findings_library import FindingsLibrary
    from clinicraft.verifier.oracle import OracleAgent

    patient = PatientHost(gtg, pack.world_config.patient)  # not invoked by oracle
    env = ClinicalEnvironment(pack, gtg, patient, FindingsLibrary.load(), mode)
    agent = OracleAgent.from_case_dir(case_dir)
    obs = await env.reset()
    turns = 0
    while not obs.episode_done and turns < 50:
        action = await agent.act(obs.model_dump(mode="json"))
        obs, done = await env.step(action)
        turns += 1
        if done:
            break
    return env, turns


def test_episode_runs_end_to_end(tmp_path):
    """#1/#2/#3: reset() + step() must not crash; oracle reaches terminal."""
    from clinicraft.schemas.interaction import PerceptionMode
    case_dir, gtg, pack = _mini_case_pack(tmp_path, PerceptionMode.STRUCTURED_ONLY)
    env, turns = asyncio.run(_run_oracle(case_dir, gtg, pack, PerceptionMode.STRUCTURED_ONLY))
    assert 0 < turns < 50
    # #6: final_submission captured
    assert "submit_diagnosis" in env.final_submission
    assert "submit_differential" in env.final_submission
    assert "submit_problem_rep" in env.final_submission


def test_frame_stream_produces_frames(tmp_path):
    """#5: frame_stream must actually render frames into the observation."""
    from clinicraft.schemas.interaction import PerceptionMode
    case_dir, gtg, pack = _mini_case_pack(tmp_path, PerceptionMode.FRAME_STREAM)

    from clinicraft.environment.clinical_env import ClinicalEnvironment
    from clinicraft.patient.host import PatientHost
    from clinicraft.physio.findings_library import FindingsLibrary
    patient = PatientHost(gtg, pack.world_config.patient)
    env = ClinicalEnvironment(pack, gtg, patient, FindingsLibrary.load(),
                              PerceptionMode.FRAME_STREAM)
    obs = asyncio.run(env.reset())
    assert obs.channels.vision is not None
    assert len(obs.channels.vision.frames) >= 1
    # In frame_stream mode the text signs must be hidden
    assert obs.channels.structured_state.visible_signs == []


def test_repeated_exam_no_crash(tmp_path):
    """#3: auscultating/inspecting the same site twice must not raise."""
    from clinicraft.schemas.interaction import PerceptionMode, Action, ActionType
    case_dir, gtg, pack = _mini_case_pack(tmp_path, PerceptionMode.STRUCTURED_ONLY)
    from clinicraft.environment.clinical_env import ClinicalEnvironment
    from clinicraft.patient.host import PatientHost
    from clinicraft.physio.findings_library import FindingsLibrary

    async def _go():
        env = ClinicalEnvironment(pack, gtg, PatientHost(gtg, pack.world_config.patient),
                                  FindingsLibrary.load(), PerceptionMode.STRUCTURED_ONLY)
        await env.reset()
        a = Action(action=ActionType.INSPECT, params={"region": "face"})
        await env.step(a)
        obs, _ = await env.step(a)  # repeat — previously crashed
        return obs
    obs = asyncio.run(_go())
    assert obs is not None


def test_fractional_vitals_coerced(tmp_path):
    """#2: mock physio decays HR to a float; Vitals(HR=int) must not crash."""
    from clinicraft.schemas.interaction import PerceptionMode, Action, ActionType
    case_dir, gtg, pack = _mini_case_pack(tmp_path, PerceptionMode.STRUCTURED_ONLY)
    from clinicraft.environment.clinical_env import ClinicalEnvironment
    from clinicraft.patient.host import PatientHost
    from clinicraft.physio.findings_library import FindingsLibrary

    async def _go():
        env = ClinicalEnvironment(pack, gtg, PatientHost(gtg, pack.world_config.patient),
                                  FindingsLibrary.load(), PerceptionMode.STRUCTURED_ONLY)
        await env.reset()
        obs, _ = await env.step(Action(action=ActionType.INSPECT, params={"region": "face"}))
        return obs
    obs = asyncio.run(_go())
    assert obs.channels.structured_state.vitals.HR is not None
    assert isinstance(obs.channels.structured_state.vitals.HR, int)
    # #10: BP must be present (derived from SBP/DBP)
    assert obs.channels.structured_state.vitals.BP is not None


# --------------------------------------------------------------------------
# Scorer (#11 renormalisation + veto) and judge veto (#8)
# --------------------------------------------------------------------------

def test_scorer_perfect_play_reaches_100():
    """#11: a rubric with only some dimensions must still allow final=100."""
    from clinicraft.judge.llm_judge import JudgeVerdict, RequirementScore
    from clinicraft.judge.scorer import compute_score
    from clinicraft.schemas.rubric import Rubric, RubricRequirement, ScoreFormula, DimensionWeights
    rubric = Rubric(case_id="S1", requirements=[
        RubricRequirement(id="r1", cat="C1", description="dx", weight=3.0),
        RubricRequirement(id="r2", cat="C2", description="rr", weight=2.0),
    ], score_formula=ScoreFormula(weights=DimensionWeights()))
    verdict = JudgeVerdict(case_id="S1", model_id="m", completeness_ok=True,
        requirement_scores=[RequirementScore(req_id="r1", score=1.0, rationale=""),
                            RequirementScore(req_id="r2", score=1.0, rationale="")])
    card = compute_score(verdict, rubric, {"total_tokens": 10})
    assert card.final == 100.0


def test_scorer_hard_veto_zeroes():
    from clinicraft.judge.llm_judge import JudgeVerdict, RequirementScore
    from clinicraft.judge.scorer import compute_score
    from clinicraft.schemas.rubric import Rubric, RubricRequirement, ScoreFormula, DimensionWeights
    rubric = Rubric(case_id="S2", requirements=[
        RubricRequirement(id="r1", cat="C1", description="dx", weight=3.0),
        RubricRequirement(id="r2", cat="C5", description="红旗", weight=3.0, veto_if_fail=True),
    ], score_formula=ScoreFormula(weights=DimensionWeights()))
    verdict = JudgeVerdict(case_id="S2", model_id="m", completeness_ok=True,
        veto_triggered=["r2"],
        requirement_scores=[RequirementScore(req_id="r1", score=1.0, rationale="")])
    card = compute_score(verdict, rubric, {})
    assert card.final == 0.0
    assert card.safety_veto is True
