"""pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_case_text():
    return """
主诉：胸痛伴大汗1小时

现病史：患者，男性，58岁，1小时前无明显诱因突发胸骨后压榨性疼痛，
疼痛放射至左肩及左臂，伴大汗淋漓、恶心，无发热，无咳嗽。
既往高血压病史10年，服用氨氯地平5mg/日。否认糖尿病、冠心病史。
吸烟史30年，约20支/日。

体格检查：
血压：90/60mmHg，心率：118次/分，呼吸：24次/分，SpO2：94%，体温：37.1°C
神志清楚，大汗，皮肤湿冷，面色苍白
心脏：心率118次/分，律齐，各瓣膜听诊区未闻及明显杂音
肺部：双肺呼吸音清，未闻及干湿啰音
腹部：腹软，无压痛

辅助检查：
ECG：窦性心动过速，V1-V4导联ST段抬高0.3-0.5mV
高敏肌钙蛋白I：2456ng/L（正常<26ng/L）

诊断：急性前壁ST段抬高型心肌梗死（STEMI）

处置：立即启动导管室，阿司匹林300mg+氯吡格雷600mg负荷，
持续心电监护，建立静脉通路，急诊PCI治疗。
"""


@pytest.fixture
def sample_gtg_dict():
    return {
        "case_id": "TEST-CARDIO-001",
        "problem_representation": "急性单系统胸痛，自主神经症状，血流动力学不稳定的58岁男性吸烟者",
        "final_dx": "前壁STEMI",
        "final_dx_icd10": "I21.0",
        "atomic_facts": ["胸痛1小时", "大汗", "血压90/60", "ECG前壁ST抬高", "肌钙蛋白升高"],
        "differential": [
            {"dx": "前壁STEMI", "p_prior": 0.65, "must_not_miss": True, "supporting": ["典型症状", "ECG"], "against": []},
            {"dx": "主动脉夹层", "p_prior": 0.10, "must_not_miss": True, "supporting": [], "against": ["无撕裂感", "ECG阳性"]},
            {"dx": "急性心包炎", "p_prior": 0.05, "must_not_miss": False, "supporting": [], "against": ["低血压", "局部导联"]},
        ],
        "must_not_miss": ["主动脉夹层"],
        "red_flags": ["血压差", "撕裂性胸痛", "神经系统症状"],
        "visible_signs": [
            {"sign_id": "pallor", "description": "皮肤苍白湿冷", "region": "face", "severity": "moderate", "render_tier": "T1", "lr_pairs": []},
            {"sign_id": "diaphoresis", "description": "大汗淋漓", "region": "face", "severity": "severe", "render_tier": "T1", "lr_pairs": []},
        ],
        "ideal_workup": [
            {"test": "ECG", "rationale": "首诊必查", "timing": "immediate", "lr_pairs": [{"dx": "STEMI", "lr_pos": 100.0, "lr_neg": 0.1, "source": "JAMA-RCE"}]},
            {"test": "troponin_I", "rationale": "确认心肌损伤", "timing": "immediate", "lr_pairs": []},
        ],
        "management_plan": [
            {"action": "急诊PCI", "rationale": "时间窗内开通梗死相关动脉", "timing": "immediate", "guideline_ref": "ACC/AHA STEMI 2013"},
        ],
        "expert_reasoning_trace": [
            "1. 中年男性吸烟者，典型胸痛+自主神经症状+血流动力学不稳 → 高危ACS",
            "2. ECG示前壁ST抬高+肌钙蛋白升高 → 确认STEMI",
            "3. 需排除主动脉夹层（血压不对称？ → 本例无）",
            "4. 后验概率>治疗阈值，立即启动导管室",
        ],
        "safety_net_items": ["若溶栓后2h胸痛未缓解或ST抬高未下降50%应补救PCI"],
        "steering_traps": [
            {"description": "患者说'只是胃痛'，不要被其引导诊断为消化道疾病"},
        ],
        "difficulty": "medium",
        "rarity": "common",
        "error_prone": False,
        "specialty": "cardiology",
        "perception_tier": "T1",
        "dynamic_coverage": 0.5,
        "validated": False,
    }
