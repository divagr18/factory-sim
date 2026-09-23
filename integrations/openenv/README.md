# factory-sim for OpenEnv

An [OpenEnv](https://github.com/huggingface/OpenEnv) environment for LLM agents.
The task is `construct_smelting_line` in factory-sim, a tick-exact simulator of
Factorio's early game. The agent must place a burner mining drill on iron ore,
put a stone furnace under its drop point, and fuel both. The reward is checked
against the simulator: a line counts only if at least 10 iron plates are
machine-made during a verification window after the build phase.

Built against `openenv` 0.5.0 (PyPI), using the WebSocket session protocol at
`/ws`. This package is `factory_sim_env`. The simulator itself (`fsim`,
`evolve`) comes from the factory-sim checkout this directory lives in.

## Two modes

**Program mode** (the default) is single-step. `reset()` returns the task, the
program contract, the `World` API reference and the game notes. This is the
same system prompt the repository's evolution loop uses. The one action is a
whole Python program, `def build(world): ...`. It can be bare source or a model
completion that contains a fenced ```python block, in which case the last block
defining `build` is used. `step()` checks the program with `evolve.sandbox`,
then runs it on the episode's scenes in a pool of worker processes, so a
program that hangs or crashes costs a worker, not the server. The episode then
ends (`done=True`).

- **reward** = the fraction of scenes on which the line was verified, in [0, 1].
- The observation also carries `family_rates`, `macro_mean`, `n_scenes`, `error`
  (a sandbox refusal, a crash, a timeout) and `failure_traces`: for each family,
  the last 20 actions of one failing episode. `text` renders all of this for an
  LLM.

**Tool mode** (`reset(mode="tool")`) is multi-turn, with one scene per episode.
Each action is one `World` call: `move`, `place`, `give`, `take`, `mine` or
`wait`, with `args` and `kwargs`, or `finish`. Calls are checked against
`World`'s signatures. Pass an entity as its integer `row` from the latest
observation. Every observation is the full published world view as JSON:
position, inventory, ore and blocked tiles, the entity table and decisions left.
A call that returns `false` was refused and cost no decision. The episode ends
on `finish()`, when the 600-decision budget runs out, or when the task times
out.

- **reward** = 1.0 if the line is verified at the end of the episode, else 0.0.
  Intermediate steps give 0. The finished episode is replayed through
  `fsim.program_api.run_episode`, the same scorer program mode uses.

## Splits, and the holdout

| split | scenes | default `n_scenes` | served by |
|---|---|---:|---|
| `train` | fresh draws, seeds in `[0, 100000)` of the train families | 16 | `server.app:app` |
| `val` | the pinned validation set of `evolve.evaluate.scene_sets()` (256) | 16 | `server.app:app` |
| `holdout` | FactorioRL's frozen holdout, 100 `obstructed_patch` scenes | 100 | `server.app:holdout_app` only |

`reset(seed=...)` picks the scenes deterministically, and `state().scenes` lists
them. If you omit the seed, the server draws one and records it in `state().seed`.

**The holdout must never provide training reward.** A training server refuses
`split="holdout"`. The holdout lives in a separate app, which you run as a
separate process on its own port, and it never returns failure traces. Point
training code at the train server only.

## Run it locally

Use Python 3.11 or later. factory-sim's C extension must be built for your
platform (`python build.py` in the repository root).

```bash
pip install "openenv>=0.5.0" cffi numpy
cd integrations/openenv
export PYTHONPATH=/path/to/factory-sim     # the server also finds it on its own
uvicorn server.app:app --port 8000          # train / val
uvicorn server.app:holdout_app --port 8001  # holdout, for evaluation only
# or: python -m server.app --port 8000 [--holdout]
```

Configuration is through environment variables:

- `FSIM_OPENENV_WORKERS`: simulator worker processes. The default is `min(4, cpus)`.
- `FSIM_OPENENV_JOB_TIMEOUT_S`: the time limit per chunk of scenes. The default is 30.
- `FSIM_OPENENV_MAX_SESSIONS`: concurrent WebSocket sessions. The default is 64.

A program is scored on 16 scenes in well under a second once the workers are up.

From Python:

```python
from factory_sim_env import FactorySimAction, FactorySimEnv

with FactorySimEnv(base_url="http://localhost:8000").sync() as env:
    r = env.reset(seed=0)                                # program mode, 16 train scenes
    print(r.observation.text)                            # the prompt
    r = env.step(FactorySimAction(program=source))
    print(r.reward, r.observation.family_rates, r.observation.error)

    r = env.reset(mode="tool", seed=3)
    r = env.step(FactorySimAction(method="move", args=["E", "long"]))
    r = env.step(FactorySimAction(method="finish"))
```

To import `factory_sim_env`, either install this directory
(`pip install -e integrations/openenv`) or run from inside it.

OpenEnv's HTTP `/reset` and `/step` routes build a fresh environment for every
request ([huggingface/OpenEnv#1156](https://github.com/huggingface/OpenEnv/issues/1156)).
A program-mode `POST /step` still scores a program on 16 train scenes, because
that step is a whole episode. Tool mode needs the WebSocket client.

### Docker

The build context is the repository root, because the image compiles the
simulator from `csrc/` with the Linux flags `-std=c11 -O2 -ffp-contract=off`:

```bash
docker build -f integrations/openenv/server/Dockerfile -t factory-sim-env .
docker run -p 8000:8000 factory-sim-env
docker run -p 8001:8000 -e FSIM_APP=holdout_app factory-sim-env
```

## Train with TRL

TRL's `GRPOTrainer(environment_factory=...)` takes a class whose public methods
become tools. `trl_envs.py` has two such classes. `ProgramToolEnv` exposes one
tool, `submit_program`. `WorldToolEnv` exposes `move`, `place`, `give`, `take`,
`mine`, `wait` and `finish`. Both keep the episode reward in `env.reward`.

```python
import os
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from factory_sim_env.trl_envs import ProgramToolEnv, reward_func

os.environ["FSIM_OPENENV_URL"] = "http://localhost:8000"   # the train server

# Each row's columns are passed to reset(). `seed` picks the scenes, so every
# generation for a prompt is scored on the same scenes.
dataset = Dataset.from_dict({
    "prompt": [[{"role": "user", "content":
        "Write a builder program for this factory and submit it with submit_program."}]] * 256,
    "seed": list(range(256)),
})

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    train_dataset=dataset,
    reward_funcs=reward_func,
    args=GRPOConfig(max_completion_length=4096, num_generations=8),
    environment_factory=ProgramToolEnv,
)
trainer.train()
```

`reset()` returns the task prompt, which TRL adds to the conversation. A
`ProgramToolEnv` episode accepts one submission; a second one raises, and TRL
passes the error back to the model. Set `FSIM_OPENENV_MAX_SESSIONS` to at least
TRL's `generation_batch_size`, because each generation opens its own session.

For a single-turn GRPO setup without tool calling, score completions directly
in a reward function: reset with a seed, then step with the completion as
`program`.

## Layout

```
openenv.yaml          manifest (openenv validate --level static passes)
pyproject.toml        package factory_sim_env; `server` entry point
models.py             FactorySimAction / FactorySimObservation / FactorySimState
client.py             FactorySimEnv (EnvClient over /ws)
trl_envs.py           TRL environment_factory wrappers
server/app.py         create_app(...) -> app, holdout_app
server/factory_sim_environment.py   the Environment classes
server/scoring.py     scene selection, sandbox + pool scoring, tool sessions
server/Dockerfile
```

Tests: `tests/test_openenv_integration.py` in the repository. It is skipped
when `openenv` is not installed.
