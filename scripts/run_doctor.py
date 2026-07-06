#!/usr/bin/env python3
"""
Run a doctor model (SUT) against a case directory.
Saves encounter trace for later scoring.

Usage:
  python scripts/run_doctor.py --model claude-opus-4-8 --case cases/neuro/MW-NEURO-01234
  python scripts/run_doctor.py --agent oracle --case cases/cardio/CASE_001
  python scripts/run_doctor.py --agent nop   --case cases/cardio/CASE_001
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from loguru import logger
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main(
    case_dir: Path = typer.Option(..., "--case", "-p", help="Path to case directory"),
    agent: str = typer.Option("", "--agent", help="oracle|nop|<model-id>"),
    model: str = typer.Option("", "--model", help="LLM model ID for doctor SUT"),
    perception: str = typer.Option("frame_stream", "--perception",
                                   help="frame_stream|structured_only|dual"),
    seed: int = typer.Option(42, "--seed"),
    output_dir: Path = typer.Option(Path("traces"), "--out"),
):
    """Run a doctor model against a CliniCraft-Bench case."""
    from clinicraft.config import settings
    from clinicraft.schemas.case_pack import CasePack
    from clinicraft.schemas.ground_truth import GroundTruthGraph
    from clinicraft.schemas.interaction import PerceptionMode

    if not case_dir.exists():
        console.print(f"[red]Case directory not found: {case_dir}[/red]")
        raise typer.Exit(1)

    pack = CasePack.load(case_dir)
    gtg = GroundTruthGraph.model_validate_json(
        (case_dir / "ground_truth_graph.json").read_text()
    )

    model_id = agent or model or settings.llm_model
    pmode = PerceptionMode(perception)

    console.print(f"\n[bold]CliniCraft-Bench Episode[/bold]")
    console.print(f"  Case:       {pack.case_id}")
    console.print(f"  Agent:      {model_id}")
    console.print(f"  Perception: {pmode.value}")

    trace = asyncio.run(_run_episode(pack, gtg, model_id, pmode, case_dir, seed))

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{pack.case_id}_{model_id.replace('/', '_')}_{seed}.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"\n[green]Trace saved: {trace_path}[/green]")
    console.print(f"  Turns: {len(trace.get('turns', []))}")
    console.print(f"  Tests ordered: {trace.get('tests_ordered', 0)}")


async def _run_episode(pack, gtg, model_id: str, pmode, case_dir: Path, seed: int) -> dict:
    """Run one full episode and return the encounter trace dict."""
    import anthropic
    from clinicraft.config import settings
    from clinicraft.environment.clinical_env import ClinicalEnvironment, EncounterTrace
    from clinicraft.patient.host import PatientHost
    from clinicraft.physio.findings_library import FindingsLibrary
    from clinicraft.verifier.oracle import NopAgent, OracleAgent

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    findings_lib = FindingsLibrary.load()

    patient = PatientHost(gtg, pack.world_config.patient, client=client)
    env = ClinicalEnvironment(pack, gtg, patient, findings_lib, pmode)
    obs = await env.reset()

    trace = EncounterTrace(
        case_id=pack.case_id,
        model_id=model_id,
        perception_mode=pmode.value,
        seed=seed,
    )

    # Select agent
    if model_id == "oracle":
        agent_obj = OracleAgent.from_case_dir(case_dir, client=client)
    elif model_id == "nop":
        agent_obj = NopAgent()
    else:
        agent_obj = _make_llm_doctor(model_id, client, pack)

    start = time.time()
    while not obs.episode_done:
        action = await agent_obj.act(obs.model_dump(mode="json"))
        trace.add_turn(obs, action)
        obs, done = await env.step(action)
        if done:
            break

    trace.wall_time_s = time.time() - start
    trace.tests_ordered = env.budget.tests_ordered
    trace.sim_cost_cny = env.budget.sim_cost_cny
    trace.final_submission = env.final_submission or None
    # Token accounting: doctor agent + patient host (judges counted at eval).
    doctor_tokens = getattr(agent_obj, "tokens", 0)
    patient_tokens = patient.tokens
    trace.total_tokens = doctor_tokens + patient_tokens

    return {k: v for k, v in trace.__dict__.items() if not k.startswith("_")}


def _make_llm_doctor(model_id: str, client, pack):
    """Wrap an LLM as a doctor agent using the §5 typed action protocol."""
    from clinicraft.schemas.interaction import Action, ActionType

    class LLMDoctorAgent:
        _SYSTEM = (
            "你是一名经验丰富的主治医师，正在对患者进行问诊。\n"
            "你的目标是：高效采集病史、有针对性地查体、合理开具检查、\n"
            "形成准确的诊断和处置计划。\n"
            "每回合选择一个动作，以JSON格式输出：\n"
            "{\"action\": \"<action_type>\", \"params\": {...}}\n"
            "可用动作：ask, auscultate, palpate, percuss, inspect, observe_task, "
            "order_test, order_imaging, prescribe, submit_problem_rep, "
            "submit_differential, submit_diagnosis, submit_plan, "
            "express_uncertainty, safety_net, escalate"
        )

        def __init__(self):
            self._history = []
            self.tokens = 0

        async def act(self, obs_dict: dict) -> Action:
            dialogue = obs_dict.get("channels", {}).get("dialogue", "")
            vitals = obs_dict.get("channels", {}).get("structured_state", {}).get("vitals", {})
            last = obs_dict.get("channels", {}).get("last_action_result", {})
            turn = obs_dict.get("turn", 1)

            self._history.append({
                "role": "user",
                "content": (
                    f"[回合{turn}]\n"
                    f"患者说：{dialogue or '（等待）'}\n"
                    f"生命体征：{json.dumps(vitals, ensure_ascii=False)}\n"
                    f"上次动作结果：{json.dumps(last, ensure_ascii=False)[:300]}\n"
                    "请选择下一步动作（JSON格式）："
                )
            })

            response = await client.messages.create(
                model=model_id,
                max_tokens=512,
                system=self._SYSTEM,
                messages=self._history[-20:],
            )
            if response.usage:
                self.tokens += response.usage.input_tokens + response.usage.output_tokens
            text = response.content[0].text.strip()
            self._history.append({"role": "assistant", "content": text})

            try:
                import re
                m = re.search(r'\{.*\}', text, re.DOTALL)
                if m:
                    data = json.loads(m.group())
                    return Action.model_validate(data)
            except Exception:
                pass
            return Action(action=ActionType.ASK, params={"utterance": "请继续描述您的症状"})

    return LLMDoctorAgent()


if __name__ == "__main__":
    app()
