# factorio-tools

### Overview
- **Environment ID**: `factorio-tools`
- **Short description**: Build a working iron smelting line in [factory-sim](https://github.com/divagr18/factory-sim) by calling `world.*` tools, one scene per rollout. The reward is verified success.
- **Tags**: tool-use, multi-turn, simulation, agent, eval

### What it tests
This is the same task as [`factorio-build`](../factorio_build/README.md), done
interactively. The model does not write a program. It calls the `world` API
one method at a time, reads each result, and calls `finish` when the line is
built and fuelled. The episode then runs its verification window. A rollout
that never calls `finish` scores 0, because its build phase never ended and
nothing was verified.

### How it works
- One task is one scene. Tasks use the same splits and seed plan as `factorio-build` (`train`, `val`, and `holdout`, which is evaluation only and must be asked for by name).
- `WorldToolset` (`servers/world.py`) is a task-scoped `vf.Toolset`, so every rollout gets its own MCP server process, simulator and scene (fetched in `setup_task`).
- The episode itself is factory-sim's `run_episode`, unchanged. `session.WorldSession` runs it on a thread whose `build(world)` takes each tool call from a queue. A tool call is one `World` method call, and `finish` returns from `build`. The budget, refusals and verification are therefore exactly a program's. The tests check that driving the seed program through the tools gives the same result, decision for decision, as running it directly.
- After every call, the server publishes the counters (and, after `finish`, the verified result) into the rollout's `WorldState`, and the reward reads it there.

Tools: `me`, `tile`, `inventory`, `ore_tiles(kind)`, `blocked_tiles`,
`entities`, `patch`, `decisions_left`, `last_refused` cost no decision.
`move(direction, stride)`, `place(item, x, y, facing)`,
`give(row, item, amount)`, `take(row, item, amount)`, `mine(row)` and
`wait(count)` spend decisions. `finish()` ends the build phase. Entities are
named by their current `row` in `entities()`.

### Rubric
| Signal | Kind | Meaning |
| --- | --- | --- |
| `success` | reward, 1.0 | 1 if `finish` was called and the line verified. |
| `finished`, `verified_output`, `decisions`, `refusals`, `tool_calls` | metrics | From the episode. |

### Quickstart
From a factory-sim checkout:

```bash
uv pip install -e . -e integrations/verifiers/factorio_build -e integrations/verifiers/factorio_tools
eval factorio-tools -m <model> -n 8 --env.agent.max-turns 400
```

A build takes 50 to 200 actions, so give the agent a generous `max_turns`. This
environment is expensive per rollout, and it is meant for evaluating agents
more than for cheap RL.

### Status and caveats
- This is a native verifiers v1 taskset only. There is no v0 `load_environment`.
- It has been tested offline: the toolset is driven directly and scored with `Task.score`, and `eval --dry-run` resolves it. It has not been run end to end with a model. That needs a model endpoint, plus the MCP server launch (`python -m factorio_tools.servers.world`) and its state channel.
- `verifiers.v1` does not import on Windows, so run it on Linux or macOS.
- For Hub packaging, see `factorio_build/README.md`. Both packages need factory-sim to be installable first.
