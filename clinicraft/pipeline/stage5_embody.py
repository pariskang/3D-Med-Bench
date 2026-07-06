"""
Stage 5: Embodiment Authoring
Maps GTG visible_signs → render parameters via Sign Rendering Library.
Selects avatar persona and assigns patient disclosure behaviour.
"""

from __future__ import annotations

from clinicraft.render.sign_library import SignRenderLibrary
from clinicraft.schemas.case_pack import PatientConfig, WorldConfig
from clinicraft.schemas.clinical_case import ClinicalCase
from clinicraft.schemas.ground_truth import GroundTruthGraph
from loguru import logger


def embody_case(
    case: ClinicalCase,
    gtg: GroundTruthGraph,
    sign_lib: SignRenderLibrary | None = None,
) -> dict:
    """
    Returns:
      render_params: list of {sign_id, blendshapes, animation, texture_overrides}
      patient_config: PatientConfig
      avatar_spec:    {model_id, age_band, sex, body_type}
      perception_tier: T1/T2/T3 (based on renderable sign coverage)
    """
    sign_lib = sign_lib or SignRenderLibrary.load()

    render_params = []
    t1_count = t2_count = t3_count = 0

    for sign in gtg.visible_signs:
        rp = sign_lib.resolve(sign.sign_id, sign.severity)
        if rp:
            render_params.append(rp)
            if sign.render_tier == "T1":
                t1_count += 1
            elif sign.render_tier == "T2":
                t2_count += 1
            else:
                t3_count += 1
        else:
            logger.warning(f"[{gtg.case_id}] No render params for sign '{sign.sign_id}'")
            t3_count += 1

    total = len(gtg.visible_signs) or 1
    if t1_count / total >= 0.8:
        tier = "T1"
    elif (t1_count + t2_count) / total >= 0.6:
        tier = "T2"
    else:
        tier = "T3"

    avatar_spec = _select_avatar(case)
    patient_config = PatientConfig(
        age=case.age,
        sex=case.sex.value if case.sex else None,
        persona=_derive_persona(gtg),
    )

    logger.info(
        f"[{gtg.case_id}] Embodiment: tier={tier}, "
        f"T1={t1_count}, T2={t2_count}, T3={t3_count}"
    )
    return {
        "render_params": render_params,
        "patient_config": patient_config.model_dump(),
        "avatar_spec": avatar_spec,
        "perception_tier": tier,
    }


def _derive_persona(gtg: GroundTruthGraph) -> str:
    """
    Derive a patient persona from the case. Anxiety-provoking or error-prone
    cases get a less cooperative persona to raise interaction difficulty.
    """
    if gtg.error_prone:
        return "anxious"
    if gtg.difficulty == "hard":
        return "dismissive"
    return "cooperative"


def _select_avatar(case: ClinicalCase) -> dict:
    age = case.age or 50
    sex = (case.sex.value if case.sex else "M").upper()

    if age < 18:
        age_band = "child"
    elif age < 40:
        age_band = "young_adult"
    elif age < 65:
        age_band = "middle_aged"
    else:
        age_band = "elderly"

    return {
        "model_id": f"cc0_{sex.lower()}_{age_band}_01",
        "age_band": age_band,
        "sex": sex,
        "body_type": "average",
    }
