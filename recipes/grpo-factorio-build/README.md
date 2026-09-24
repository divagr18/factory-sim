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
