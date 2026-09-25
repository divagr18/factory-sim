"""factorio-build for the classic (v0) verifiers stack: `load_environment(**kwargs)`.

verifiers 0.3.x ships both stacks. Its v1 `eval` resolves the exported
`FactorioBuildTaskset`; its `vf-eval`, `prime env init` templates and v0
trainers call `load_environment`. Both score through `core.score_completion`,
so a completion earns the same reward on either.
"""

from __future__ import annotations

import verifiers as vf  # the v0 surface (verifiers.legacy)

from factorio_build import core

ENV_ID = "factorio-build"


def _text(parser, completion) -> str | None:
    if isinstance(completion, str):
        return completion
    return parser.parse_answer(completion)


def load_environment(
    task: str = "construct_smelting_line",
    n_scenes: int = 16,
    split: str = "train",
    game_notes: bool = True,
    prompt_version: str = "v1",
    num_examples: int = 64,
    seed: int = 0,
    workers: int = 4,
    job_timeout_s: float = 30.0,
    decision_budget: int | None = None,
    format_weight: float = 0.1,
    refusal_weight: float = 0.0,
    **kwargs,
) -> vf.SingleTurnEnv:
    """A single-turn program-synthesis environment over factory-sim scenes.

    `split="holdout"` is the frozen held-out set and is only built when asked for
    by name; never train on it."""
    from datasets import Dataset

    rows = core.rows(task, split, n_scenes, num_examples, seed, game_notes, prompt_version)
    system = rows[0]["system_prompt"]
    dataset = Dataset.from_list(
        [
            {
                "question": r["prompt"],
                "answer": "",
                "task": ENV_ID,
                "info": {
                    "subset_id": r["subset_id"],
                    "split": r["split"],
                    "sim_task": r["task"],
                    "scenes": [
                        {"family": f, "seed": s, "sample_split": ss} for f, s, ss in r["scenes"]
                    ],
                },
            }
            for r in rows
        ]
    )
    parser = vf.Parser()

    def scored(completion, info, state) -> dict:
        # Reward functions run one after another per rollout, so the first one
        # computes and the rest read the cached, JSON-serialisable result.
        cached = state.get("factorio_build")
        if cached is None:
            scenes = [(s["family"], s["seed"], s["sample_split"]) for s in info["scenes"]]
            cached = core.score_completion(
                _text(parser, completion),
                info["sim_task"],
                scenes,
                workers=workers,
                timeout_s=job_timeout_s,
                decision_budget=decision_budget,
            )
            state["factorio_build"] = cached
        return cached

    async def success_rate(completion, info, state, **_) -> float:
        import asyncio

        return (await asyncio.to_thread(scored, completion, info, state))["success"]

    async def format(completion, info, state, **_) -> float:
        return core.format_score(scored(completion, info, state))

    async def refusal_penalty(completion, info, state, **_) -> float:
        return -scored(completion, info, state)["refusal_rate"]

    rubric = vf.Rubric(
        funcs=[success_rate, format, refusal_penalty],
        weights=[1.0, format_weight, refusal_weight],
        parser=parser,
    )
    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system,
        parser=parser,
        rubric=rubric,
        env_id=ENV_ID,
        **kwargs,
    )
