#!/usr/bin/env bash
# One-shot setup of prime-rl + factorio-build on a fresh GPU pod (tested on
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, 2x A100 80GB).
#
#   ROOT=/root MODELS="Qwen/Qwen3.5-9B" bash pod_setup.sh [factorio_build-*.whl]
#
# Idempotent: re-running skips finished steps. Point ROOT at a network volume
# (e.g. /workspace) to keep the venv, checkouts and model weights across pods.
set -euo pipefail

ROOT=${ROOT:-/root}
MODELS=${MODELS:-Qwen/Qwen3.5-9B}
PRIME_RL_REF=${PRIME_RL_REF:-main}
WHEEL=${1:-}
export HF_HOME=${HF_HOME:-$ROOT/hf}
LOG=$ROOT/setup.log
mkdir -p "$ROOT"
# A fresh log per run, so waiters never match an earlier run's errors.
if [ -f "$LOG" ]; then mv "$LOG" "$LOG.prev"; fi
exec > >(tee "$LOG") 2>&1
step() { echo; echo "=== $* ($(date +%T))"; }

step "uv"
# Always install the current uv: some images ship an old /usr/bin/uv that cannot
# parse prime-rl's uv.lock ("invalid type: boolean `false`, expected a timestamp").
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

step "prime-rl checkout"
# Some submodules are declared with git@github.com: URLs; the pod has no GitHub key.
git config --global url."https://github.com/".insteadOf git@github.com:
export GIT_TERMINAL_PROMPT=0
if [ ! -d "$ROOT/prime-rl/.git" ]; then
    git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$ROOT/prime-rl"
fi
cd "$ROOT/prime-rl"
git fetch -q origin "$PRIME_RL_REF" && git checkout -q FETCH_HEAD
SUBS="deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config"
git submodule update --init --force -- $SUBS
# A failed first clone can leave a submodule with only its .git file.
for s in $SUBS; do
    [ -f "$s/pyproject.toml" ] || (cd "$s" && git checkout -f HEAD)
    [ -f "$s/pyproject.toml" ] || { echo "submodule $s is empty"; exit 1; }
done

step "uv sync (the slow part: torch, vLLM, flash-attn)"
uv sync --all-extras

step "factory-sim + factorio-build"
# prime-rl's pyproject sets `exclude-newer = "7 days"`, which hides a factory-sim
# release newer than a week; pin past it explicitly.
uv pip install --exclude-newer "$(date -u -d tomorrow +%Y-%m-%dT00:00:00Z)" "factory-sim>=0.1.2"
if [ -n "$WHEEL" ]; then
    # --no-deps: the wheel pins verifiers>=0.3.1, the workspace verifiers is a dev build.
    uv pip install --no-deps --force-reinstall "$WHEEL"
else
    uv pip install --no-deps "git+https://github.com/divagr18/factory-sim@main#subdirectory=integrations/verifiers/factorio_build"
fi
# For the SFT warm start; prime-rl itself never imports these.
uv pip install --no-deps peft accelerate
.venv/bin/python -c "import factorio_build, fsim, peft; assert factorio_build.V1_IMPORT_ERROR is None; print('envs ok')"

step "models"
for m in $MODELS; do
    uvx --from "huggingface_hub[hf_xet]" hf download "$m" >/dev/null
    echo "downloaded $m"
done

step "helper scripts"
cat > "$ROOT/stop_rl.sh" <<'EOF'
#!/usr/bin/env bash
# Kills a prime-rl run. Run it as a script: `pkill -f` inside an inline
# `ssh host '...'` matches the ssh command line itself and kills the session.
for p in "rl @" prime_rl torchrun EngineCore vllm spawn_main; do pkill -f "$p" || true; done
sleep 12
pkill -9 -f spawn_main || true; pkill -9 -f EngineCore || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader
EOF
cat > "$ROOT/start_rl.sh" <<EOF
#!/usr/bin/env bash
# start_rl.sh <config.toml> [extra rl args...]: detached, survives the ssh session.
export PATH="\$HOME/.local/bin:\$PATH"
cd $ROOT/prime-rl
export HF_HOME=$HF_HOME FACTORIO_BUILD_MAX_POOLS=\${FACTORIO_BUILD_MAX_POOLS:-7}
cfg=\$1; shift
setsid nohup uv run --no-sync rl @ "\$cfg" "\$@" > $ROOT/rl.log 2>&1 < /dev/null &
echo "started; tail -f $ROOT/rl.log, then outputs/*/logs/attempt_1/{orchestrator,trainer}.log"
EOF
cat > "$ROOT/serve.sh" <<EOF
#!/usr/bin/env bash
# serve.sh <model> [gpu]: detached vLLM server on :8000.
export PATH="\$HOME/.local/bin:\$PATH"
cd $ROOT/prime-rl
export HF_HOME=$HF_HOME
CUDA_VISIBLE_DEVICES=\${2:-0} setsid nohup uv run --no-sync inference --vllm.model "\$1" > $ROOT/serve.log 2>&1 < /dev/null &
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done; echo ready
EOF
chmod +x "$ROOT"/{stop_rl,start_rl,serve}.sh

step "done"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "SETUP_DONE"
