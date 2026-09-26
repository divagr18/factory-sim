# Recipe: SFT warm start + GRPO on factorio-build

Train Qwen3.5-9B to write `def build(world):` programs with
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), on 2× A100 80GB
(one GPU runs vLLM, one runs the trainer).

## Why a warm start

Base Qwen3.5-4B and 9B solve ~0% of validation scenes: half their programs
exceed the sandbox's size limit and most of the rest fail to parse. GRPO learns
nothing from groups where every rollout scores the same, so the model first
imitates programs found by `evolve`, then RL takes over.

## Steps

```bash
# laptop: pod up, then
MODELS="Qwen/Qwen3.5-9B" bash recipes/grpo-factorio-build/deploy.sh <ip> <port>
uv run python recipes/grpo-factorio-build/build_sft_data.py \
    --archives "runs/*/genealogy.sqlite" --target 800 --out sft.jsonl
scp sft.jsonl rp:/root/recipe/

# pod
cd /root/prime-rl
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python /root/recipe/sft_lora.py \
    --data /root/recipe/sft.jsonl --out /root/models/sft            # ~10 min per 100 programs x 2 epochs
/root/serve.sh /root/models/sft                                      # eval before RL
uv run --no-sync eval @ /root/recipe/configs/eval.toml --model /root/models/sft
/root/stop_rl.sh
/root/start_rl.sh /root/recipe/configs/rl.toml                        # GRPO
```

Evaluate at temperature 0.6 with thinking off. Use `split = "holdout"` only for
final numbers.

## Results

Qwen3.5-9B on `construct_smelting_line`, scored in the simulator: 16 rows x 4
samples x 8 scenes per split, temperature 0.6, thinking off. PlanBench is
Blocksworld plan generation (task 1) and plan execution (task 7), 500
instances each, thinking off.

| Qwen3.5-9B | `val` | `holdout` | PlanBench generation | PlanBench execution |
|---|---|---|---|---|
| base | 2.1% | 1.6% | 53.2 | 39.8 |
| SFT-800 + GRPO (Steps above) | 85.9% | 84.0% | 2.8 | 3.0 |
| Evolve & Reinforce, round 1 (below) | 85.6% | 71.9% | 46.4 | 27.8 |

Evolve & Reinforce is the self-voiced warm start below (`rationalize.py`),
with replay (`make_replay.py`), then GRPO. Models:
[`qwen3.5-9b-factorio-build-er-r1`](https://huggingface.co/divagr1925/qwen3.5-9b-factorio-build-er-r1)
and its warm start
[`qwen3.5-9b-factorio-build-er-r1-sft`](https://huggingface.co/divagr1925/qwen3.5-9b-factorio-build-er-r1-sft).
All models: [models collection].

## Warm start in the model's own voice

The SFT-800 warm start above, followed by GRPO, took Qwen3.5-9B from 1.6% to
84.0% on the holdout. It also cost the model its general planning: PlanBench
Blocksworld fell from 53% to 3% with thinking off, and replies shrank from
~2,000 tokens to ~60. SFT caused the whole drop. The targets were bare
programs written by another model. Training on targets in the model's own
voice forgets less (STaR's rationalisation, self-distillation fine-tuning,
RL's Razor), and one round of self-SFT still forgets unless it is iterated
(Retaining by Doing). These scripts rebuild the SFT data that way.

| File | What it does |
|---|---|
| `rationalize.py` | Base Qwen writes each target: the env's prompt, plus one message holding the evolved program, becomes an explanation and a program. Only replies whose program scores at least as well as the evolved one are kept. The SFT row is the prompt without the hint. |
| `make_replay.py` | Base Qwen's own answers to general prompts, mixed into SFT as replay. |
| `sft_lora.py` | New flags: `--empty-think`, `--replay`, `--replay-frac`. A local `--model` starts a second round. |
| `sft_data.py` | Examples, loss masks and the replay mix for `sft_lora.py`. It needs no torch, so the tests use it. |
| `gen_common.py` | The async client for vLLM and the resumable JSONL store that both generators use. |

### What these scripts decide

- **Hint** (`HINT_TEMPLATE`, version `hint-v1`, written to `run.json` and `stats.json`).
  The request is the env's system and user messages unchanged, then one more
  user message:

  ````text
  A reference program that solves this task:

  ```python
  {program}```

  Explain the approach step by step in your own words, as your own solution, without mentioning the reference. Then give the complete program in one ```python fenced block, with nothing after it. You may adapt or improve the reference program.
  ````

  "Without mentioning the reference" is there because the target is trained
  against the prompt without the hint. The examples are asked and trained
  with the env's `prompt_version` v2, which also asks for step-by-step prose
  before the program, so a reply that reasons does not contradict its prompt
  (v1 asks for a plan of at most 5 lines). That prose is the habit SFT removed.
- **Prompts are the env's.** Each example's scenes are rebuilt from the subset
  id in its prompt. `core.rows` must reproduce its system and user messages
  byte for byte, or the run stops before sending any request. Only train
  scenes are accepted. All 800 SFT-800 rows match the current env.
- **Scoring is the env's.** `evolve.llm.extract_code` extracts the program and
  `factorio_build.core.score_completion` scores it, on the example's own scenes,
  with the env's defaults (4 workers, 30 s timeout, the task's decision budget).
- **Keep rule**, per sample. The request succeeded, the reply finished, and it
  holds a sandbox-valid program. Its success on the example's scenes is at least
  the reference program's, and above zero. The reply does not mention the hint,
  holds one program block, and has at least 30 words of prose before it. Of the
  samples that pass, the best one per example is kept: higher success first,
  then the prose closest to the median length. `--rebuild` re-applies other keep settings
  (`--keep-per-example`, `--min-reasoning-words`, `--max-programs`,
  `--allow-zero-success`) without regenerating.
- **Empty think block.** With `enable_thinking=False`, Qwen3.5's generation
  prompt ends with `<think>\n\n</think>\n\n`: four tokens, `<think>`, `\n\n`,
  `</think>`, `\n\n`. The old `sft_lora.py` already kept them out of the loss,
  because it tokenized that generation prompt as the prompt. So the SFT-800
  drop did not come from training on the empty block. `--empty-think mask`,
  the default, keeps those labels exactly as before. `--empty-think train`
  puts the four tokens in the loss, as trainers that take the loss over the
  whole rendered assistant turn do. That is the ablation.
- **Replay sources and licences.** OpenAssistant/oasst2 (Apache-2.0): English
  first turns only, excluding deleted, synthetic, spam, PII and inappropriate
  prompts. openai/gsm8k (MIT): the train split. The default mix is 75/25, 1,000
  prompts, and 20% of them are answered with thinking on, keeping the
  reasoning. Planning look-alikes (Blocksworld, stacking blocks, PDDL,
  PlanBench's phrasing) are dropped, and the counts land in `selection.json`.
  Replay always comes from base, in every round.
- **Replay share.** `--replay-frac 0.2` makes replay 20% of each epoch: 200
  replay rows next to 800 task rows. Replay is drawn without replacement across
  epochs. Without `--replay`, the example order is the old one.

### Runbook

Before you start, check the pod's env imports. `deploy.sh` packs factory-sim
and builds the factorio-build wheel from the same committed HEAD, and
`pod_setup.sh` installs both; without `FSIM_SRC` it falls back to
`factory-sim>=0.2.0` from PyPI. Uncommitted changes are not deployed.

```bash
ssh rp 'cd /root/prime-rl && .venv/bin/python -c "from factorio_build import core; print(core.SUPPORTED_TASKS)"'
```

```bash
# laptop: pod up, recipe and wheel uploaded, SFT-800 data built (see Steps)
MODELS="Qwen/Qwen3.5-9B" bash recipes/grpo-factorio-build/deploy.sh <ip> <port>
uv run python recipes/grpo-factorio-build/build_sft_data.py \
    --archives "runs/*/genealogy.sqlite" --target 800 --out sft.jsonl
scp sft.jsonl rp:/root/recipe/

# pod: 1. serve BASE on GPU 0 (thinking is switched per request)
cd /root/prime-rl
export HF_HOME=/root/hf
/root/serve.sh Qwen/Qwen3.5-9B 0                                     # prints "ready"

# 2. rationalise: an 8-example smoke test, then all 800 (the same --out-dir resumes)
uv run --no-sync python /root/recipe/rationalize.py \
    --data /root/recipe/sft.jsonl --out-dir /root/rat_r1 --limit 8
cat /root/rat_r1/stats.json            # kept rate, reject reasons, lengths
setsid nohup uv run --no-sync python /root/recipe/rationalize.py \
    --data /root/recipe/sft.jsonl --out-dir /root/rat_r1 \
    > /root/rat_r1.log 2>&1 < /dev/null &
tail -f /root/rat_r1.log               # rerun the same command after any drop

# 3. replay from base, on the same server
setsid nohup uv run --no-sync python /root/recipe/make_replay.py \
    --out-dir /root/replay --n-prompts 1000 --think-frac 0.2 \
    > /root/replay.log 2>&1 < /dev/null &
/root/stop_rl.sh                       # when both logs end with "kept ..."

# 4. SFT (sft_lora.py merges the LoRA into /root/models/sft itself)
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python /root/recipe/sft_lora.py \
    --data /root/rat_r1/sft.jsonl --replay /root/replay/replay.jsonl --replay-frac 0.2 \
    --out /root/models/sft

# 5. eval before RL, as in Steps
/root/serve.sh /root/models/sft
uv run --no-sync eval @ /root/recipe/configs/eval.toml --model /root/models/sft
/root/stop_rl.sh

# 6. GRPO, then merge the adapter you keep into a full checkpoint
/root/start_rl.sh /root/recipe/configs/rl.toml
uv run --no-sync python /root/recipe/merge_lora.py --base /root/models/sft \
    --adapter outputs/<run>/broadcasts/step_20 --out /root/models/grpo
```

Check `stats.json` after the smoke test:
- `replies_with_reasoning_content` must be 0. Otherwise the server's
  reasoning parser is moving reply text out of `content`; serve without one.
- `reject_reasons` shows why samples fail. `truncated` means `--max-tokens` is
  too low, and `hint_leak` means the replies talk about the hint.
- `kept_vs_hint` gives the share of kept programs identical to the hint
  (exactly, or up to names and comments) and the share that differ.

**Second round (Iterative-SFT).** Regenerate from the current checkpoint
instead of base, then train on from it. `--hint-mode fallback` samples each
example without the hint first. It adds hinted samples only where no unhinted
one passes, which is STaR's order. The replay stays from base.

```bash
/root/serve.sh /root/models/sft 0
setsid nohup uv run --no-sync python /root/recipe/rationalize.py \
    --data /root/recipe/sft.jsonl --out-dir /root/rat_r2 --round 2 --hint-mode fallback \
    > /root/rat_r2.log 2>&1 < /dev/null &
/root/stop_rl.sh                       # when the log ends with "kept ..."
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python /root/recipe/sft_lora.py --model /root/models/sft \
    --data /root/rat_r2/sft.jsonl --replay /root/replay/replay.jsonl --replay-frac 0.2 \
    --epochs 1 --out /root/models/sft_r2
/root/start_rl.sh /root/recipe/configs/rl.toml --model.name /root/models/sft_r2
```

### Time and cost of the generation stage (1× A100 80GB, $1.59/hr)

These are estimates, not measurements. The token counts are measured with the
Qwen3.5 tokenizer on the SFT-800 file: 2,960 prompt tokens with the hint, and
1,550 for the program alone. The reply length is assumed: the program plus
500-2,000 tokens of prose. Throughput is assumed to be 3,000-5,000 generated
tokens/s with 256 sequences in flight (64 requests x 4 samples). For
comparison, the GRPO numbers above work out to at least ~3,800 generated
tokens/s: 128 rollouts of ~4k tokens (~1.3k of them prompt), generated and
scored in ~1.5 min on the one inference A100, with a LoRA adapter.

| Stage | Work | Time | Cost |
|---|---|---|---|
| vLLM start | load, CUDA graphs | ~5 min | $0.13 |
| `rationalize.py`, round 1 | 800 requests x 4 samples, 6-11M generated tokens; 2.4M prompt tokens, prefilled once per request | 25-65 min | $0.65-1.75 |
| scoring | 3,200 programs x 8 scenes, ~0.2 s CPU each, overlapped with generation | ~0 extra | - |
| `make_replay.py` | 1,000 prompts, ~1-3M tokens; the longest thinking-on replies set the end | 10-20 min | $0.25-0.55 |
| **generation total** | | **~40-90 min** | **~$1.05-2.40** |
| `sft_lora.py` (for scale) | if ~700 rows are kept: ~1.3-1.7x the old rows' length, plus 20% replay; scaled from the 10 min per 100 programs x 2 epochs above | ~1.5-2.5 h | ~$2.40-4.00 |

A second round with `--hint-mode fallback` sends one unhinted request per
example, plus a hinted one for each example the checkpoint fails unhinted. It
costs between round 1 and twice round 1.

## What broke, and what the scripts do about it

| Symptom | Cause | Fix (where) |
|---|---|---|
| submodule clone asks for credentials | submodules declared with `git@github.com:` | rewrite to https (`pod_setup.sh`) |
| `deps/pydantic-config does not appear to be a Python project` | a failed clone left an empty submodule | force checkout and verify each (`pod_setup.sh`) |
| `no versions of factory-sim` | prime-rl sets `exclude-newer = "7 days"` | explicit `--exclude-newer` (`pod_setup.sh`) |
| `hf: command not found`, pip PEP 668 error | no system `hf`, externally managed Python | `uvx --from huggingface_hub hf` (`pod_setup.sh`) |
| server or eval dies when ssh drops | process tied to the session | `setsid nohup` (`start_rl.sh`, `serve.sh`) |
| ssh session kills itself on stop | inline `pkill -f` matches the ssh command line | stop from a script file (`stop_rl.sh`) |
| 5-17 episodes in flight, ~12 min/step | concurrency sized for the 262k context | `[orchestrator.concurrency]` (`rl.toml`) |
| hundreds of rollouts "buffered", one CPU at 100% | every rollout scored through one pool | `FACTORIO_BUILD_MAX_POOLS` (factorio-build `core.py`) |
| RL warns LoRA without inference | no `[inference]` section | present (`rl.toml`) |
| eval samples think for thousands of tokens | Qwen3.5-9B defaults to thinking | `chat_template_kwargs` (`eval.toml`), renderer (`rl.toml`) |
| SFT model 0% at temperature 1.0, 18% at 0.6 | long programs drift when sampled hot | train at 0.7, evaluate at 0.6 |

## Throughput (2× A100 80GB, 9B, LoRA r16)

- Trainer: ~2.5 min/step for 128 rollouts (~2.7k tokens/s, ~43% MFU). This is the bottleneck.
- Rollouts + scoring: ~1.5 min per 128 with 4 workers × 7 pools.
- Warm-up before step 1: ~5 min (vLLM load + CUDA graphs + weight broadcast).
