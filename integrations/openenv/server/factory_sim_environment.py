"""The factory-sim OpenEnv environment: build a smelting line, get a verified reward.

Program mode (default): `reset` returns the task prompt; the one action is a
whole `def build(world)` program; `step` sandbox-checks it, runs it on the
episode's scenes (16 by default) and returns the success rate as the reward,
with `done=True`.

Tool mode (`reset(mode="tool")`): one scene; every action is one `World` call;
the reward is 1 on a verified line at the end of the episode, else 0.

`FactorySimEnvironment` serves the train and val splits. The frozen holdout is
served only by `FactorySimHoldoutEnvironment`, which a separate app mounts
(`server.app:holdout_app`), so a training server cannot hand out holdout reward
by accident, and holdout scoring never returns a failure trace.
"""

from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import FactorySimAction, FactorySimObservation, FactorySimState
    from . import scoring
except ImportError:  # run from the environment root: `uvicorn server.app:app`
    from models import FactorySimAction, FactorySimObservation, FactorySimState
    from server import scoring

MODES = ("program", "tool")


class FactorySimEnvironment(
    Environment[FactorySimAction, FactorySimObservation, FactorySimState]
):
    """construct_smelting_line on factory-sim, for LLM agents."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    SPLITS: tuple[str, ...] = scoring.TRAIN_SPLITS

    def __init__(self) -> None:
        super().__init__()
        self._state = FactorySimState(episode_id=str(uuid4()), split=self.SPLITS[0])
        self._scenes: list[tuple[str, int, dict]] = []
        self._tool: scoring.ToolSession | None = None
        self._ready = False

    # ------------------------------------------------------------ openenv API

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        mode: str = "program",
        split: str | None = None,
        n_scenes: int | None = None,
        **kwargs: Any,
    ) -> FactorySimObservation:
        split = split or self.SPLITS[0]
        if split not in self.SPLITS:
            raise ValueError(
                f"split {split!r} is not served here; this server serves {list(self.SPLITS)}"
                + (" (the holdout has its own server: server.app:holdout_app)"
                   if split == "holdout" else "")
            )
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "tool":
            n = 1
        else:
            n = scoring.DEFAULT_SCENES[split] if n_scenes is None else int(n_scenes)
            if n < 1:
                raise ValueError("n_scenes must be at least 1")
        if seed is None:
            seed = random.randrange(2**31)
        self._scenes = scoring.select_scenes(split, n, int(seed))
        scene_list = [{"family": f, "seed": s} for f, s, _ in self._scenes]
        self._state = FactorySimState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            mode=mode,
            split=split,
            seed=int(seed),
            task=scoring.TASK,
            scenes=scene_list,
        )
        self._ready = True
        if mode == "tool":
            family, scene_seed, blueprint = self._scenes[0]
            self._tool = scoring.ToolSession(family, scene_seed, blueprint)
            prompt = scoring.tool_prompt()
            view = self._tool.view()
            return FactorySimObservation(
                mode=mode, split=split, prompt=prompt, api_reference=scoring.api_reference(),
                scenes=scene_list, world=view,
                text=prompt + "\n\nWorld:\n" + scoring.dumps(view),
                done=False, reward=0.0,
            )
        self._tool = None
        prompt = scoring.program_prompt()
        return FactorySimObservation(
            mode=mode, split=split, prompt=prompt, api_reference=scoring.api_reference(),
            scenes=scene_list, n_scenes=n,
            text=prompt + f"\n\nYour program will be run on {n} {split} scenes.",
            done=False, reward=0.0,
        )

    def step(
        self,
        action: FactorySimAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> FactorySimObservation:
        if not self._ready:
            # The HTTP /step route builds a fresh environment per request; a
            # program-mode step is a whole episode, so it still means something.
            self.reset(mode="program" if action.program is not None else "tool")
        st = self._state
        if st.done:
            error = "the episode is over; call reset()"
            return self._obs(text=f"Error: {error}", error=error, done=True, reward=0.0)
        st.step_count += 1
        if st.mode == "program":
            return self._program_step(action)
        return self._tool_step(action)

    @property
    def state(self) -> FactorySimState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="factory_sim",
            description=(
                "Build an iron smelting line in factory-sim, a tick-exact simulator of "
                "Factorio's early game. Program mode scores one Python builder program on "
                "many scenes; tool mode plays one scene a World call at a time. "
                f"Splits served: {', '.join(self.SPLITS)}."
            ),
            version="0.1.0",
            documentation_url="https://github.com/divagr18/factory-sim",
        )

    def close(self) -> None:
        self._tool = None

    # ------------------------------------------------------------ modes

    def _program_step(self, action: FactorySimAction) -> FactorySimObservation:
        st = self._state
        if action.program is None:
            return self._finish(0.0, error="program mode takes `program`, not `method`")
        result = scoring.score_program(action.program, self._scenes, st.split)
        return self._finish(
            result["reward"],
            text=scoring.program_feedback(result, st.split),
            success_rate=result["success"],
            family_rates=result["rates"],
            macro_mean=result["mean"],
            n_scenes=result["n"],
            failure_traces=result["traces"],
            error=result["error"],
        )

    def _tool_step(self, action: FactorySimAction) -> FactorySimObservation:
        session = self._tool
        if action.method is None:
            error = "tool mode takes `method`, not `program`"
            return self._obs(text=f"Error: {error}", error=error, world=session.view(),
                             done=False, reward=0.0)
        ok, error = session.call(action.method, action.args, action.kwargs)
        self._state.decisions = session.world.decisions
        view = session.view()
        if session.done:
            out = session.outcome
            reward = 1.0 if out["success"] else 0.0
            text = (
                f"Episode over: {'verified' if out['success'] else 'not verified'}, "
                f"{out['verified_output']} plates in the verification window, "
                f"{out['decisions']} decisions, {out['refusals']} refusals."
            )
            return self._finish(reward, text=text, world=view, result=ok,
                                verified_output=out["verified_output"],
                                error=error or out["error"])
        if error is not None:
            text = f"Error: {error}"
        else:
            text = f"{action.method} -> {'ok' if ok else 'refused'}"
            if ok and view["last_refused"]:
                text = f"{action.method} -> failed (the game refused it; a decision was spent)"
        return self._obs(text=text + "\n\nWorld:\n" + scoring.dumps(view), world=view,
                         result=ok, error=error, done=False, reward=0.0)

    # ------------------------------------------------------------ helpers

    def _obs(self, **fields: Any) -> FactorySimObservation:
        st = self._state
        return FactorySimObservation(mode=st.mode, split=st.split, scenes=st.scenes, **fields)

    def _finish(self, reward: float, **fields: Any) -> FactorySimObservation:
        st = self._state
        st.done = True
        st.reward = float(reward)
        if "text" not in fields:
            fields["text"] = f"Error: {fields.get('error')}"
        return self._obs(done=True, reward=float(reward), **fields)


class FactorySimHoldoutEnvironment(FactorySimEnvironment):
    """Evaluation only: FactorioRL's frozen holdout (100 `obstructed_patch` scenes).

    Its reward is a measurement. Never train on it."""

    SPLITS = scoring.HOLDOUT_SPLITS
