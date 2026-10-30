# CORE (NON-NEGOTIABLE)

1) Simplicity first
- Write the simplest correct code.
- Use straightforward control flow.
- Minimize lines of code. Fewer lines is a goal.
- Keep answers and explanations as SHORT and SIMPLE as possible.

2) Validate every change
- Run the narrowest check that proves it; never `make ci` mid-task, full suite once before committing.
- Report the exact command + output. No "should work" without a run log.

3) No safety nets
- Do NOT use try/except.
- Do NOT use hasattr.
- Do NOT add fallbacks or alternate paths.
- Assume inputs and environment are correct.
- Let the program crash on errors.

4) Function structure
- Prefer fewer, larger functions.
- Do NOT split into many small helper functions.
- Only create a new function when it removes real duplication and still keeps the total function count low.

5) Real robot
- NEVER run scripts that move the real robot (deploy, park, calibration captures). Prepare the command and let the human run it.
- Read-only hardware access (camera streams, joint-state reads, device enumeration) is fine.

# Notes

- Use `uv` to run scripts (not `python` or `pip` directly).
- GPU scripts: find a free GPU first and set `CUDA_VISIBLE_DEVICES`. Visualization scripts with infinite loops need a timeout.
- Type hints on function args and return types (except `None`, `Any`, `object`).
- Do NOT add `from __future__ import annotations` to any file.
- Do NOT add any Claude/AI attribution or trailers to commit messages.

# Project Overview

`graspkit` — dexterous manipulation research: grasp optimization, motion planning, simulation, policy learning, and real-robot deployment.

# Common Commands

```shell
# Install / sync (ManiSkill and spconv are extras; the full pipeline needs Python 3.10–3.12)
uv sync --extra maniskill --extra spconv

# Lint, format and test
make ci             # format + lint + test
make format         # uvx ruff format
make lint           # uvx ruff check --fix
make test           # uv run pytest tests (set CUDA_VISIBLE_DEVICES to a free GPU)
make test-oldest    # Python 3.10 at the oldest allowed versions

# Run any script
uv run scripts/<path>.py [args]

# Full pipeline (optimize → plan → render → process → train)
uv run scripts/launch.py <experiment_name>
# Experiments defined in src/graspkit/configs/experiment/
```

# Architecture

```
src/graspkit/
  constants.py        # PROJECT_ROOT, DATA_ROOT, LOG_ROOT, S3/HF paths
  assets/             # graspkit HF-dataset access (fetch/link + scenes/objects modules)
  configs/            # All config classes (*_config.py, tyro CLI); instances apart: experiment/, robot/, grasp_policy.py, mj_hand.py, real_env.py
  models/             # GraspPolicy, generators, vision encoders, grasp evaluator
  datasets/           # HDF5/msgpack dataset loaders
  stages/             # Pipeline stages: grasp_optim/ (optimizers, warp GD/LM helpers, whole-body), motion_gen/, trainer
  robot/              # GraspRobot (FK/IK + contact-point facade), RobotHelper
  scene/              # Scene descriptions and geometry (SceneDescription, SceneGeom)
  envs/               # BaseEnv, build_env/build_renderer and the ManiSkill, MuJoCo, Isaac and real adapters
  mani_skill/         # ManiSkill3 agents (15+ robot hands), envs and the training-data renderer
  isaac/              # Isaac simulator implementation and isolated runtime worker
  utils/              # Transforms, point clouds, cameras, visualization
  viser/              # 3D interactive visualization
scripts/
  launch.py           # Unified multi-stage pipeline launcher
  stages/             # Per-stage scripts (grasp_optim, motion_gen, data_gen, obs_gen, train, ...)
  benchmark/          # Benchmarks (grasp_opt/ grid: preset x backend x variant vs reference md)
  deploy/             # Real-robot deployment (xARM, G1)
  tools/              # Asset/visualization tools
tests/                # Pytest suite mirroring src/graspkit/ and scripts/ (tests/scripts/); shared fixtures in conftest.py
assets/               # graspkit HF-dataset checkout via graspkit.assets.link(); not git-tracked
```

# Key Conventions

- **Config system**: All configs are dataclasses parsed via `tyro`. Configs live in `src/graspkit/configs/`. Scripts accept CLI args that map to config fields.
- **data/ and logs/ are symlinks** to `~/.graspkit/{data,logs}`, auto-created on import. This allows sharing across git worktrees. Never commit real data into the repo.
- **Robot assets** are consumed per-robot from robokit (`robokit.assets.robots.end_effectors.<name>.<SYMBOL>`), which downloads only that robot on import from HF (`openrobokit/robokit`) into `~/.robokit/assets/<pinned-commit>/`. graspkit's own assets (scenes, objects) come from `graspkit.assets`. `import graspkit` downloads nothing. See `docs/reference/data/assets.md`.
- **SE(3) transforms**: Follow robokit conventions. Use `robokit.xform` for rotation conversions and SE(3) operations.
- **robokit boundary**: robokit holds robot-generic math (terms, solvers, lie, geom, robo); graspkit holds research pipelines and anything tuned per-experiment (all of grasp-opt lives in `stages/grasp_optim/`). Grasp code enters robokit only by graduation — moved, never copied — when a second consumer exists.
- **robokit dependency**: consumed as a pinned git rev in `pyproject.toml [tool.uv.sources]`. To pick up robokit changes: merge there, bump the rev, `uv sync`.
- **Robot hands**: Each hand has a config in `configs/robot/`, a ManiSkill agent in `mani_skill/agents/`, and URDF assets exposed as symbols in `robokit.assets.robots.end_effectors.<name>`.
- **Git workflow**: Feature branches with squash-merge to `main`. Pre-commit hooks run ruff. GitHub CI runs ruff and the CPU test suite, where tests marked `cuda` or `hf_assets` skip; run the full suite on a GPU with Hugging Face access before merging (`CUDA_VISIBLE_DEVICES=0 uv run pytest tests`).
