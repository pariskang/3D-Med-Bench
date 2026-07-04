"""
ClinicalEnvironment — Layer 4, §5 interaction protocol.

Manages the typed observation-action loop:
  1. Build Observation from current state
  2. Receive Action from SUT (doctor model)
  3. Execute action (update patient state, advance physio, re-render)
  4. Record EncounterTrace for replay and scoring
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from clinicraft.patient.host import PatientHost
from clinicraft.physio.findings_library import FindingsLibrary
from clinicraft.physio.pulse_client import get_pulse_client
from clinicraft.render.scene_renderer import get_scene_renderer
from clinicraft.schemas.case_pack import CasePack
from clinicraft.schemas.ground_truth import GroundTruthGraph
from clinicraft.schemas.interaction import (
    Action, ActionResult, ActionType, AudioChannel,
    Budget, Channel, Observation, PerceptionMode,
    StructuredState, VisionChannel, Vitals,
)


@dataclass
class EncounterTrace:
    """Full record of one doctor-patient encounter. Used for replay and scoring."""
    case_id: str
    model_id: str
    perception_mode: str
    seed: int
    turns: list[dict[str, Any]] = field(default_factory=list)
    final_submission: dict[str, Any] | None = None
    wall_time_s: float = 0.0
    total_tokens: int = 0
    tests_ordered: int = 0
    sim_cost_cny: float = 0.0
    terminated_reason: str = ""

    def add_turn(self, obs: Observation, action: Action) -> None:
        self.turns.append({
            "turn": obs.turn,
            "observation": obs.model_dump(mode="json"),
            "action": action.model_dump(mode="json"),
            "ts": time.time(),
        })

    def save(self, out_path: Path) -> None:
        out_path.write_text(
            json.dumps(
                {k: v for k, v in self.__dict__.items()
                 if not k.startswith("_")},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )


class ClinicalEnvironment:
    """
    Seeded, deterministic clinical environment.
    One environment instance = one episode for one case.
    """

    _EXAM_FINDINGS_DB: dict[str, str] = {
        # site → finding string (from GTG visible_signs / findings_lib)
    }

    def __init__(
        self,
        pack: CasePack,
        gtg: GroundTruthGraph,
        patient: PatientHost,
        findings_lib: FindingsLibrary,
        perception_mode: PerceptionMode = PerceptionMode.FRAME_STREAM,
    ) -> None:
        self._pack = pack
        self._gtg = gtg
        self._patient = patient
        self._findings_lib = findings_lib
        self._perception_mode = perception_mode
        self._turn = 0
        self._elapsed_sim_min = 0
        self._budget = Budget()
        self._physio_client = get_pulse_client(pack.world_config.physio.scenario_id)
        self._vitals: dict[str, Any] = dict(
            pack.world_config.physio.initial_state
        )
        self._exam_results: dict[str, Any] = {}
        self._test_results: dict[str, Any] = {}
        self._events: list[dict] = []
        self._final_submission: dict[str, Any] = {}
        self._renderer = get_scene_renderer()
        self._done = False

    @property
    def final_submission(self) -> dict[str, Any]:
        return self._final_submission

    @property
    def budget(self) -> Budget:
        return self._budget

    async def reset(self) -> Observation:
        """Initialise physio engine and return first observation."""
        physio = self._pack.world_config.physio
        await self._physio_client.initialise(
            physio.scenario_id or physio.scenario_xml or "",
            physio.initial_state,
        )
        # Sync starting vitals from the engine's baseline so turn-1 reflects it.
        state = await self._physio_client.get_state()
        self._vitals.update({k: v for k, v in state.items()
                             if k in ("HR", "SBP", "DBP", "RR", "SpO2", "T", "GCS")})
        self._turn = 1   # first observation is turn 1 (not 0)
        return self._build_observation(last_result=None)

    async def step(self, action: Action) -> tuple[Observation, bool]:
        """Execute one action, advance state, return next observation + done flag."""
        result = await self._execute_action(action)

        # Capture terminal cognitive submissions for the judge / replay.
        if action.action in (ActionType.SUBMIT_DIAGNOSIS, ActionType.SUBMIT_PLAN,
                             ActionType.SUBMIT_DIFFERENTIAL, ActionType.SUBMIT_PROBLEM_REP):
            self._final_submission[action.action.value] = action.params

        # Advance physio by 1 sim-minute per turn
        self._elapsed_sim_min += 1
        state = await self._physio_client.advance(dt_seconds=60.0)
        self._vitals.update({k: v for k, v in state.items()
                             if k in ("HR", "SBP", "DBP", "RR", "SpO2", "T", "GCS")})

        done = self._check_done(action)
        self._turn += 1
        obs = self._build_observation(result)
        if done:
            self._done = True
            obs = obs.model_copy(update={"episode_done": True})
        return obs, done

    async def _execute_action(self, action: Action) -> ActionResult:
        atype = action.action
        params = action.params

        if atype == ActionType.ASK:
            response = await self._patient.respond(params.get("utterance", ""))
            return ActionResult(action=atype.value, finding=response)

        elif atype in (ActionType.AUSCULTATE, ActionType.PALPATE,
                        ActionType.PERCUSS, ActionType.CHECK_REFLEX,
                        ActionType.CHECK_PULSE, ActionType.INSPECT,
                        ActionType.OBSERVE_TASK, ActionType.FUNDOSCOPY,
                        ActionType.OPHTHALMOSCOPY,
                        # §12 望闻切 map to the exam resolver (望/闻/切 signs)
                        ActionType.TCM_INSPECT, ActionType.TCM_LISTEN,
                        ActionType.TCM_PULSE):
            return self._execute_exam(atype, params)

        elif atype == ActionType.ORDER_TEST:
            return self._execute_test(params.get("test", ""))

        elif atype == ActionType.ORDER_IMAGING:
            return self._execute_imaging(params.get("study", ""))

        elif atype in (ActionType.PRESCRIBE, ActionType.ESCALATE):
            await self._physio_client.apply_action(
                atype.value, params
            )
            return ActionResult(action=atype.value, finding="已记录处置指令")

        elif atype in (ActionType.SUBMIT_DIAGNOSIS, ActionType.SUBMIT_PLAN,
                        ActionType.SUBMIT_DIFFERENTIAL, ActionType.SUBMIT_PROBLEM_REP,
                        ActionType.EXPRESS_UNCERTAINTY, ActionType.CHOOSE_NEXT_STEP):
            return ActionResult(action=atype.value, finding="已记录认知动作")

        elif atype == ActionType.SAFETY_NET:
            return ActionResult(action=atype.value, finding="安全网指令已记录")

        return ActionResult(action=atype.value, error="未知动作类型")

    def _execute_exam(self, atype: ActionType, params: dict) -> ActionResult:
        """Look up physical exam finding from GTG visible_signs / findings_lib."""
        site = params.get("site") or params.get("region") or params.get("task", "")
        key = f"{atype.value}:{site}"

        # Try cached result first (determinism). The cache stores the full
        # ActionResult minus 'action'; rebuild by injecting action only.
        if key in self._exam_results:
            cached = dict(self._exam_results[key])
            return ActionResult(action=atype.value, **cached)

        # Resolve from GTG visible signs
        for sign in self._gtg.visible_signs:
            if site.lower() in sign.description.lower() or site.lower() in sign.region.lower():
                lr = {p.dx: p.lr_pos for p in sign.lr_pairs if p.lr_pos}
                result = ActionResult(
                    action=atype.value,
                    site=site,
                    finding=sign.description,
                    lr_pairs=[lr] if lr else [],
                )
                self._exam_results[key] = result.model_dump(exclude={"action"})
                return result

        # Not in GTG → return normal finding
        result = ActionResult(
            action=atype.value, site=site,
            finding=f"{site} 检查未见明显异常",
        )
        self._exam_results[key] = result.model_dump(exclude={"action"})
        return result

    def _execute_test(self, test_name: str) -> ActionResult:
        if test_name in self._test_results:
            return ActionResult(action="order_test", site=test_name,
                                finding=self._test_results[test_name])
        entry = self._findings_lib.resolve(test_name, self._elapsed_sim_min)
        if entry:
            finding = entry.result_value or entry.result_template
            lr = list(entry.lr_pairs.items())[:1]
            lr_pairs = [{dx: vals.get("lr_pos", 1.0)} for dx, vals in lr]
        else:
            finding = f"{test_name}：结果待出（未在本案例脚本中定义）"
            lr_pairs = []
        self._test_results[test_name] = finding
        self._budget.tests_ordered += 1
        self._budget.sim_cost_cny += 50.0  # default test cost
        return ActionResult(action="order_test", site=test_name,
                            finding=finding, lr_pairs=lr_pairs)

    def _execute_imaging(self, study: str) -> ActionResult:
        return self._execute_test(study)  # same resolution path

    def _build_observation(self, last_result: ActionResult | None) -> Observation:
        # In frame_stream mode the model must perceive signs from the rendered
        # frames; in structured_only/dual they are also given as text.
        show_text_signs = self._perception_mode != PerceptionMode.FRAME_STREAM
        vis_signs = [s.description for s in self._gtg.visible_signs] if show_text_signs else []

        state = StructuredState(
            vitals=self._build_vitals(),
            visible_signs=vis_signs,
        )

        # Render frames for frame_stream / dual modes.
        vision = None
        if self._perception_mode in (PerceptionMode.FRAME_STREAM, PerceptionMode.DUAL):
            frames = self._renderer.render(
                self._gtg.visible_signs,
                self._pack.world_config.patient,
                view="patient_front",
            )
            vision = VisionChannel(frames=frames, fps=4.0, view="patient_front")

        return Observation(
            turn=self._turn,
            case_id=self._gtg.case_id,
            perception_mode=self._perception_mode,
            channels=Channel(
                structured_state=state,
                last_action_result=last_result,
                vision=vision,
            ),
            available_actions=[a.value for a in ActionType],
            clock={"sim_minutes_elapsed": self._elapsed_sim_min},
            budget=self._budget,
        )

    def _build_vitals(self) -> Vitals:
        """Compose a Vitals object, deriving BP from SBP/DBP and coercing ints."""
        v = self._vitals

        def _as_int(key: str) -> int | None:
            val = v.get(key)
            return int(round(val)) if isinstance(val, (int, float)) else None

        bp = v.get("BP")
        if not bp and v.get("SBP") is not None and v.get("DBP") is not None:
            bp = f"{int(round(v['SBP']))}/{int(round(v['DBP']))}"

        spo2 = v.get("SpO2")
        temp = v.get("T")
        return Vitals(
            HR=_as_int("HR"),
            BP=bp,
            RR=_as_int("RR"),
            SpO2=float(spo2) if isinstance(spo2, (int, float)) else None,
            T=float(temp) if isinstance(temp, (int, float)) else None,
            GCS=_as_int("GCS"),
        )

    def _check_done(self, action: Action) -> bool:
        terminal = {ActionType.SUBMIT_DIAGNOSIS, ActionType.SUBMIT_PLAN}
        if action.action in terminal:
            return True
        if self._turn >= self._pack.world_config.max_turns:
            return True
        return False
