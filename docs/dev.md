# Development Guide

## Setup

GraspKit needs Python 3.10–3.14.

```shell
git clone https://github.com/openrobokit/graspkit.git
cd graspkit
uv sync --extra maniskill --extra spconv
```

`uv sync` installs the `dev` and `pt-cu126` groups by default; the full pipeline also needs the `maniskill` and
`spconv` extras (Python 3.10–3.12). For another PyTorch build, see
[Installation](guide/installation.md#pytorch-builds).

## Git Workflow

Create your own branch for development. Use pull requests with **squash merge** to merge to `main`.
Delete your branch after merging. GitHub CI runs ruff and the tests on a CPU runner without
Hugging Face access, where tests marked `cuda` or `hf_assets` skip, so run the full suite locally on a
GPU before opening a pull request.

## Formatting, Linting, and Tests

```shell
make ci                       # format + lint + test
make format                   # uvx ruff@0.15.22 format
make lint                     # uvx ruff@0.15.22 check --fix
CUDA_VISIBLE_DEVICES=0 make test  # uv run pytest tests
```

`tests/` mirrors `src/graspkit/` and `scripts/` (`tests/scripts/`), one `Test<Feature>` class per feature;
shared fixtures live in `tests/conftest.py`. Mark tests that need a GPU with `@pytest.mark.cuda` and tests
that need Hugging Face assets (robokit robots, graspkit scenes) with `@pytest.mark.hf_assets`;
import such assets inside the test, since robokit downloads a robot when its module is imported.

Pre-commit hooks run ruff. Pin the same ruff version the Makefile does; a newer default rule set
reports findings CI does not.

## Running Things

- Use `uv run`, never a bare `python` or `pip`.
- GPU scripts: pick a free GPU first and set `CUDA_VISIBLE_DEVICES`.
- Viser scripts serve until interrupted; give them a timeout when scripting them.

## Configs

All config **classes** live in `src/graspkit/configs/*_config.py`; their **instances** live apart, in
`configs/experiment/`, `configs/robot/`, `configs/grasp_policy.py`, `configs/mj_hand.py` and
`configs/real_env.py`. A new robot or experiment is a module plus one line in the registry of its
package's `__init__.py`. See the [Config System](reference/config_system.md).

## Code Style

- Type hints on function arguments and return types, except `None`, `Any` and `object`.
- No `from __future__ import annotations`.
- Prefer fewer, larger functions; add a helper only when it removes real duplication.
- No try/except, no `hasattr`, no fallback paths: assume the environment is correct and let the
  program crash on errors.

## The robokit Boundary

robokit holds robot-generic math (terms, solvers, lie, geom, robo); graspkit holds the research
pipelines and anything tuned per experiment. Code moves to robokit only when a second consumer
exists, and it is moved, never copied. robokit is pinned by git rev in `pyproject.toml`
(`[tool.uv.sources]`): to pick up robokit changes, merge there, bump the rev, then `uv sync`.
