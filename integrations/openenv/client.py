"""WebSocket client for the factory-sim environment.

    from factory_sim_env import FactorySimAction, FactorySimEnv

    with FactorySimEnv(base_url="http://localhost:8000").sync() as env:
        r = env.reset(seed=0)                        # program mode, 16 train scenes
        r = env.step(FactorySimAction(program=source))
        print(r.reward, r.observation.family_rates)

        r = env.reset(mode="tool", seed=3)           # one scene, one World call per step
        r = env.step(FactorySimAction(method="move", args=["E", "long"]))
"""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import FactorySimAction, FactorySimObservation, FactorySimState
except ImportError:
    from models import FactorySimAction, FactorySimObservation, FactorySimState


class FactorySimEnv(EnvClient[FactorySimAction, FactorySimObservation, FactorySimState]):
    """Client for `server.app:app` (train/val) or `server.app:holdout_app`."""

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        # Scoring a program on the full holdout takes a few seconds; leave room.
        kwargs.setdefault("message_timeout_s", 300.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: FactorySimAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[FactorySimObservation]:
        obs = dict(payload.get("observation", {}))
        obs["done"] = payload.get("done", False)
        obs["reward"] = payload.get("reward")
        observation = FactorySimObservation.model_validate(obs)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> FactorySimState:
        return FactorySimState.model_validate(payload)
