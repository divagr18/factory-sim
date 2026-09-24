#!/usr/bin/env bash
# From the laptop: prepare a fresh pod in one command.
#
#   bash deploy.sh <ip> <port> [alias]          # values from the pod's ssh.direct
#
# Builds the factorio-build wheel from this checkout (so unreleased fixes ship),
# uploads it with this recipe, runs pod_setup.sh detached (it survives ssh drops)
# and waits for it to finish. Pass MODELS / ROOT through the environment.
set -euo pipefail

IP=$1; PORT=$2; ALIAS=${3:-rp}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ROOT:-/root}
MODELS=${MODELS:-Qwen/Qwen3.5-9B}

# 1. ssh alias (replaces an older block with the same name)
CFG=~/.ssh/config
touch "$CFG"
python - "$CFG" "$ALIAS" "$IP" "$PORT" <<'EOF'
import re, sys
path, alias, ip, port = sys.argv[1:]
text = open(path).read()
text = re.sub(rf"\n?Host {re.escape(alias)}\n(?:[ \t]+.*\n?)*", "\n", text)
text = text.rstrip() + f"""

Host {alias}
  HostName {ip}
  Port {port}
  User root
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking accept-new
  ServerAliveInterval 30
"""
open(path, "w").write(text)
EOF
ssh -o ConnectTimeout=20 "$ALIAS" true

# 2. wheel from the working tree
rm -rf "$HERE/.dist"
(cd "$REPO/integrations/verifiers/factorio_build" && uv build --wheel -q -o "$HERE/.dist")
WHL=$(ls "$HERE"/.dist/factorio_build-*.whl)

# 3. upload recipe + wheel
ssh "$ALIAS" "mkdir -p $ROOT/recipe"
scp -q -r "$HERE/configs" "$HERE"/*.sh "$HERE"/*.py "$WHL" "$ALIAS:$ROOT/recipe/"

# 4. run setup detached, then wait on its log
ssh "$ALIAS" "cd $ROOT/recipe && ROOT=$ROOT MODELS='$MODELS' setsid nohup bash pod_setup.sh $ROOT/recipe/$(basename "$WHL") > /dev/null 2>&1 < /dev/null &"
echo "setup running on $ALIAS; log at $ROOT/setup.log"
until ssh -o ConnectTimeout=20 "$ALIAS" "grep -q 'SETUP_DONE\|^Traceback\|error:' $ROOT/setup.log 2>/dev/null"; do sleep 20; done
ssh "$ALIAS" "tail -4 $ROOT/setup.log"
