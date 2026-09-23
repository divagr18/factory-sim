"""FastAPI apps for the factory-sim environment.

Two apps, two processes, so training and evaluation never share a server:

    app          train and val splits (training reward)
    holdout_app  FactorioRL's frozen holdout (evaluation only)

Both speak OpenEnv's WebSocket session protocol at /ws (use `FactorySimEnv`),
plus /health, /schema and /metadata. OpenEnv's HTTP /reset and /step build a
throwaway environment per request (huggingface/OpenEnv#1156), so they can only
score a single program-mode step; use the client.

    cd integrations/openenv
    uvicorn server.app:app --port 8000
    uvicorn server.app:holdout_app --port 8001

Environment variables: FSIM_OPENENV_MAX_SESSIONS (default 64),
FSIM_OPENENV_WORKERS (simulator worker processes, default min(4, cpus)),
FSIM_OPENENV_JOB_TIMEOUT_S (per chunk of scenes, default 30).
"""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

try:
    from ..models import FactorySimAction, FactorySimObservation, FactorySimState
    from .factory_sim_environment import FactorySimEnvironment, FactorySimHoldoutEnvironment
except ImportError:  # run from the environment root
    from models import FactorySimAction, FactorySimObservation, FactorySimState
    from server.factory_sim_environment import (
        FactorySimEnvironment,
        FactorySimHoldoutEnvironment,
    )

MAX_SESSIONS = int(os.environ.get("FSIM_OPENENV_MAX_SESSIONS", "64"))

app = create_app(
    FactorySimEnvironment,
    FactorySimAction,
    FactorySimObservation,
    env_name="factory_sim",
    max_concurrent_envs=MAX_SESSIONS,
    state_cls=FactorySimState,
)

holdout_app = create_app(
    FactorySimHoldoutEnvironment,
    FactorySimAction,
    FactorySimObservation,
    env_name="factory_sim_holdout",
    max_concurrent_envs=MAX_SESSIONS,
    state_cls=FactorySimState,
)


def main(argv: list[str] | None = None) -> None:
    """`uv run --project . server [--host H] [--port N] [--holdout]`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the factory-sim OpenEnv environment.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--holdout", action="store_true", help="serve the frozen holdout instead")
    args = parser.parse_args(argv)
    uvicorn.run(holdout_app if args.holdout else app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
