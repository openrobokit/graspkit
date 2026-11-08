# AGENTS.md

HuG: whole-body grasping on a Unitree G1 with Inspire hands, in SAPIEN and on hardware. The product is the
entry points in `scripts/sim/` and `scripts/real/`; `src/hug` is the stack they share.

## Rules

- Write the simplest correct code, in as few lines as possible. Prefer fewer, larger functions; add a
  helper only when it removes real duplication.
- After every change, run what you changed and report the exact command and output. No "should work".
- Real-robot code keeps its `try`/`finally` and `except KeyboardInterrupt` shutdowns: Ctrl+C must leave the robot
  compliant.
- Keep explanations short.
- No Claude/AI attribution or trailers in commit messages.

## Commands

```shell
make install       # uv sync --extra sim --extra dev  (CUDA=real-cu128 / real-cu126 adds the real stack)
make ci            # ruff format-check + ruff lint + mypy
make format        # ruff format (rewrites files)
make demo-grasp    # sim grasp demo, viser on :8080
make demo-teleop   # sim teleop demo, viser on :8080
uv run --extra sim python <script>                       # sim: no torch
uv run --extra real-cu128 python <script>                # real: adds SAM2, nvdiffrast, RealSense
```

- Run everything through `uv`, never bare `python` or `pip`.
- Sync at most one torch extra (`real-cu128`, `real-cu126`); uv refuses two. Sim never needs torch.
- viser scripts never exit: run them under `timeout` when checking they start.

## Conventions

- Quaternions are wxyz; pose7 is `(x, y, z, qw, qx, qy, qz)`. scipy is xyzw: convert at the boundary and
  keep `wxyz` in the variable name.
- Transforms are `T_dst_src` (maps `v_src` to `v_dst`); joint positions are `q`.
- The G1 has 29 motors (12 legs, 3 waist, 14 arms). FK uses a 32-joint URDF with 3 head joints after the
  waist: convert with `wbc.pose_estimator.motor_q_29_to_32`, never by hand.
- Sim and real share structure, not tuning. A value tuned per target is a field on both paired configs in
  `control/config.py` (`SimControlConfig`/`RealControlConfig`, `SimGraspTiming`/`RealGraspTiming`),
  never a literal on one side.
- `hug.paths` owns every location: assets, checkpoints, everything under `results/`.
- Package `__init__.py` files stay import-light.
- Modules whose docstring says "lifted from" are surgical ports: don't restyle them.

## Style

- Type-hint arguments and returns. No `from __future__ import annotations`.
- Comments are one line. Docstrings are one line (three at most); module docstrings five lines at most;
  script docstrings twelve at most, ending in `Usage:`.
- Document a dataclass field with one trailing `# comment` (tyro shows it in `--help`). A script's CLI is a
  tyro `Args` dataclass; library tuning is a `*Config`.
- A log line starts with a lowercase subsystem tag (`[init]`, `[g1]`, `[segment]`) and says what happened, with the
  number that matters. Silence library output (banners, progress bars, per-request logs) once, at its source.
