"""factorio-tools: the factory-sim `world` API as tools, one scene per rollout.

A native verifiers v1 taskset only. verifiers.v1 needs `fcntl`, so on Windows
the taskset is not exported; `factorio_tools.session` still imports.
"""

try:
    from factorio_tools.taskset import FactorioToolsHarness, FactorioToolsTaskset
except ImportError as e:  # verifiers.v1 unavailable on this platform
    V1_IMPORT_ERROR: ImportError | None = e
    __all__: list[str] = []
else:
    V1_IMPORT_ERROR = None
    __all__ = ["FactorioToolsHarness", "FactorioToolsTaskset"]
