"""Typed wire models for the factory-sim OpenEnv environment.

One action type serves both modes, and exactly one of its two halves is set:

- program mode: `program` is a whole builder program, `def build(world): ...`,
  either as bare source or as a model completion that contains a fenced
  ```python block (the last block defining `build` is used).
- tool mode: `method` is one `World` action (`move`, `place`, `give`, `take`,
  `mine`, `wait`) or `finish`, called with `args` / `kwargs`.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from openenv.core.env_server.types import Action, Observation, State


class FactorySimAction(Action):
    """A builder program (program mode) or one `World` method call (tool mode)."""

    program: str | None = Field(
        default=None,
        description="Program mode: the source of `def build(world): ...`, bare or inside a "
        "```python fenced block.",
    )
    method: str | None = Field(
        default=None,
        description="Tool mode: move | place | give | take | mine | wait | finish.",
    )
    args: list[Any] = Field(
        default_factory=list, description="Tool mode: positional arguments for `method`."
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Tool mode: keyword arguments for `method`."
    )

    @model_validator(mode="after")
    def _one_half(self) -> FactorySimAction:
        if (self.program is None) == (self.method is None):
            raise ValueError("set exactly one of `program` (program mode) or `method` (tool mode)")
        return self


class FactorySimObservation(Observation):
    """What the agent sees. `text` is always a complete LLM-facing rendering."""

    mode: str = Field(default="program", description="program | tool")
    split: str = Field(default="train", description="Scene split the episode draws from.")
    text: str = Field(
        default="",
        description="LLM-facing text: the task prompt on reset, feedback after a step.",
    )
    prompt: str = Field(
        default="",
        description="The task description, program contract, World API and game notes "
        "(set on reset).",
    )
    api_reference: str = Field(default="", description="The World API as text (set on reset).")
    scenes: list[dict[str, Any]] = Field(
        default_factory=list, description="The episode's scenes: [{family, seed}]."
    )
    # --- program mode results ---
    success_rate: float | None = Field(
        default=None, description="Fraction of scenes on which the built line was verified."
    )
    family_rates: dict[str, float] = Field(
        default_factory=dict, description="Success rate per scene family."
    )
    macro_mean: float | None = Field(
        default=None, description="Mean of the per-family rates."
    )
    n_scenes: int = Field(default=0, description="Scenes the program was scored on.")
    failure_traces: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per family, the last actions of one failing episode (never on holdout).",
    )
    # --- tool mode ---
    world: dict[str, Any] | None = Field(
        default=None, description="Tool mode: the published world view as JSON."
    )
    result: bool | None = Field(
        default=None,
        description="Tool mode: the World method's return value "
        "(False = refused, no decision spent).",
    )
    verified_output: int | None = Field(
        default=None,
        description="Tool mode, at episode end: plates made in the verification window.",
    )
    # --- both ---
    error: str | None = Field(
        default=None, description="Why the action was rejected or failed, if it was."
    )


class FactorySimState(State):
    """Server-side episode bookkeeping."""

    mode: str = "program"
    split: str = "train"
    seed: int | None = None
    task: str = "construct_smelting_line"
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    decisions: int = 0
    done: bool = False
    reward: float | None = None
