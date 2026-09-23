"""TRL `environment_factory` wrappers for the factory-sim server.

TRL's GRPOTrainer builds one of these per generation, calls `reset(**row)`,
exposes every other public method as a tool, and reads `env.reward` in the
reward function. Nothing here imports TRL.

    from trl import GRPOConfig, GRPOTrainer
    from factory_sim_env.trl_envs import ProgramToolEnv, reward_func

    trainer = GRPOTrainer(model=..., train_dataset=dataset, reward_funcs=reward_func,
                          args=GRPOConfig(...), environment_factory=ProgramToolEnv)

The server URL comes from FSIM_OPENENV_URL (default http://localhost:8000).
Point it at the train server; never at the holdout server.
"""

from __future__ import annotations

import os

try:
    from .client import FactorySimEnv
    from .models import FactorySimAction
except ImportError:
    from client import FactorySimEnv
    from models import FactorySimAction


def _url() -> str:
    return os.environ.get("FSIM_OPENENV_URL", "http://localhost:8000")


class _Base:
    MODE = "program"

    def __init__(self) -> None:
        self._client = None
        self.reward = 0.0
        self.done = False

    def _connect(self):
        if self._client is None:
            self._client = FactorySimEnv(base_url=_url()).sync()
            self._client.connect()
        return self._client

    def reset(self, **kwargs) -> str | None:
        self.reward, self.done = 0.0, False
        options = {k: kwargs[k] for k in ("seed", "split", "n_scenes") if k in kwargs}
        result = self._connect().reset(mode=self.MODE, **options)
        return result.observation.text

    def _close(self) -> None:  # private: TRL exposes every public method as a tool
        if self._client is not None:
            self._client.close()
            self._client = None


class ProgramToolEnv(_Base):
    """Program mode: one tool, `submit_program`. The reward is the success rate."""

    MODE = "program"

    def submit_program(self, program: str) -> str:
        """
        Run a builder program on this episode's scenes and report how it did.

        Args:
            program: Python source defining `def build(world):`, bare or in a ```python block.

        Returns:
            The success rate, per-family rates, any error, and traces of failing episodes.
        """
        if self.done:
            raise ValueError("The program was already submitted; the episode is over.")
        result = self._connect().step(FactorySimAction(program=program))
        self.reward = float(result.reward or 0.0)
        self.done = True
        return result.observation.text


class WorldToolEnv(_Base):
    """Tool mode: one tool per World action. The reward is 1 on a verified line."""

    MODE = "tool"

    def _call(self, method: str, *args) -> str:
        if self.done:
            raise ValueError("The episode is over.")
        result = self._connect().step(FactorySimAction(method=method, args=list(args)))
        self.done = bool(result.done)
        if self.done:
            self.reward = float(result.reward or 0.0)
        obs = result.observation
        if obs.error and not self.done:
            raise ValueError(obs.text)
        return obs.text

    def move(self, direction: str, stride: str) -> str:
        """
        Walk one decision in a direction.

        Args:
            direction: One of N, E, S, W.
            stride: long (about 4.5 tiles), step (about 1 tile) or nudge (about 0.3 tiles).

        Returns:
            The outcome and the new world view as JSON.
        """
        return self._call("move", direction, stride)

    def place(self, item: str, x: int, y: int, facing: str) -> str:
        """
        Place an item from the inventory on a tile within 5 tiles of the character.

        Args:
            item: Item name, e.g. burner-mining-drill or stone-furnace.
            x: Tile x; a 2x2 machine covers x..x+1.
            y: Tile y; a 2x2 machine covers y..y+1.
            facing: One of N, E, S, W.

        Returns:
            The outcome and the new world view as JSON.
        """
        return self._call("place", item, x, y, facing)

    def give(self, entity: int, item: str, amount: int) -> str:
        """
        Move items from the inventory into an entity.

        Args:
            entity: The entity's row in the latest world view.
            item: Item name, e.g. coal.
            amount: 1, 5 or 20.

        Returns:
            The outcome and the new world view as JSON.
        """
        return self._call("give", entity, item, amount)

    def take(self, entity: int, item: str, amount: int) -> str:
        """
        Move items out of an entity into the inventory.

        Args:
            entity: The entity's row in the latest world view.
            item: Item name, e.g. iron-plate.
            amount: 1, 5 or 20.

        Returns:
            The outcome and the new world view as JSON.
        """
        return self._call("take", entity, item, amount)

    def mine(self, entity: int) -> str:
        """
        Pick up an entity.

        Args:
            entity: The entity's row in the latest world view.

        Returns:
            The outcome and the new world view as JSON.
        """
        return self._call("mine", entity)

    def wait(self) -> str:
        """
        Let one decision (30 ticks) pass.

        Returns:
            The outcome and the new world view as JSON.
        """
        return self._call("wait")

    def finish(self) -> str:
        """
        End the build phase; the episode runs to its end and is scored.

        Returns:
            Whether the smelting line was verified.
        """
        return self._call("finish")


def reward_func(environments, **kwargs) -> list[float]:
    """GRPO reward: each environment's episode reward."""
    return [env.reward for env in environments]
