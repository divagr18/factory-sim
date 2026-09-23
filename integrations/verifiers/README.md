# Prime Intellect `verifiers` environments

There are two packages, one environment each, in the Environments Hub layout
(a package with its own `pyproject.toml` and README):

| Package | Kind | Entry points |
| --- | --- | --- |
| [`factorio_build/`](factorio_build/README.md) | single-turn program synthesis | v1 `FactorioBuildTaskset`, v0 `load_environment()` |
| [`factorio_tools/`](factorio_tools/README.md) | multi-turn tool use, one scene per rollout | v1 `FactorioToolsTaskset` |

Both reuse factory-sim's evaluation code (`fsim.program_api`,
`evolve.sandbox`, `evolve.evaluate`, `evolve.mutate`, `fsim.scenes`). Neither
has been published to the Hub. `factorio_build/README.md` ("Packaging for the
Hub") explains what publishing requires.

Tests: `tests/test_verifiers_integration.py`. They are skipped when
`verifiers` is not installed, and the v1 cases are also skipped on Windows,
where `verifiers.v1` cannot import.

```bash
uv venv --python 3.13 .venv-verifiers
uv pip install --python .venv-verifiers -e . verifiers datasets pytest
.venv-verifiers/bin/python -m pytest tests/test_verifiers_integration.py   # Scripts\python on Windows
```
