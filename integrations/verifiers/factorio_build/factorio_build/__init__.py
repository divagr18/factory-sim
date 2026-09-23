"""factorio-build: write a `def build(world):` program that builds a working
smelting line in factory-sim, scored by running it on a subset of scenes.

`FactorioBuildTaskset` is the native verifiers v1 entry point (`eval
factorio-build`); `load_environment` is the v0 one (`vf-eval factorio-build`).

verifiers.v1 does not import on Windows (it needs `fcntl`), so there only the
v0 entry point is exported.
"""


def load_environment(**kwargs):
    """The v0 `SingleTurnEnv` (see `factorio_build.legacy.load_environment`)."""
    from factorio_build.legacy import load_environment as load

    return load(**kwargs)


try:
    from factorio_build.taskset import FactorioBuildHarness, FactorioBuildTaskset
except ImportError as e:  # verifiers.v1 unavailable on this platform
    V1_IMPORT_ERROR: ImportError | None = e
    __all__ = ["load_environment"]
else:
    V1_IMPORT_ERROR = None
    __all__ = ["FactorioBuildHarness", "FactorioBuildTaskset", "load_environment"]
