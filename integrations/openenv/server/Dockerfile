# factory-sim as an OpenEnv environment.
#
# The build context is the factory-sim repository root, since the image
# compiles the simulator from csrc/:
#
#   docker build -f integrations/openenv/server/Dockerfile -t factory-sim-env .
#   docker run -p 8000:8000 factory-sim-env                     # train/val server
#   docker run -p 8001:8000 -e FSIM_APP=holdout_app factory-sim-env   # holdout (eval only)
#
# A plain python base rather than openenv-base: the simulator needs a C
# compiler at build time, and nothing else from the base image.

FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "openenv>=0.5.0" "cffi>=1.17" "numpy>=2" setuptools

WORKDIR /app/factory-sim

# The simulator and the evolution harness (sandbox, evaluator, pool, prompts).
COPY build_fsim.py pyproject.toml LICENSE NOTICE ./
COPY csrc ./csrc
COPY fsim ./fsim
COPY evolve ./evolve
COPY integrations/openenv ./integrations/openenv
# The benchmark map's water. Without it fsim.water_tiles() silently returns []
# and every scene would be simulated on a map with no water.
COPY tests/golden/sim-mechanics-m3.json ./tests/golden/sim-mechanics-m3.json

# Linux flags from build_fsim.py: -std=c11 -O2 -ffp-contract=off. Drop the Windows
# extension copied from a developer checkout, build, and check it imports.
RUN rm -f fsim/_fsim*.pyd && python build_fsim.py && python -c "import fsim; assert fsim.water_tiles(), 'no water'; from fsim.rl import RlEnv; RlEnv()"

ENV PYTHONPATH=/app/factory-sim \
    PYTHONUNBUFFERED=1 \
    FSIM_APP=app \
    FSIM_OPENENV_WORKERS=4

WORKDIR /app/factory-sim/integrations/openenv
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:${FSIM_APP} --host 0.0.0.0 --port 8000"]
