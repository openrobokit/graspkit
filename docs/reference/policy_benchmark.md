# Grasp policy benchmark

Training, packaging, evaluation and reporting run from this GraspKit checkout. The
pipeline covers `arx_x5`, `panda_allegro`, `agilex_aloha`, `yam` and `xarm7_sharpa`,
with depth/RGB observations, CVAE/flow-transformer proposals, open/closed control
and ManiSkill/Isaac/MuJoCo backends.

Experiment artifacts belong under `graspkit.constants.DATA_ROOT`, normally
`~/.graspkit/data`. The archived 2026-10-13 run keeps its original name,
`DATA_ROOT/runs/five_robot_checkpoints_20261013` (S3 copy under the same name).
Training defaults to `DATA_ROOT/runs/grasp_policy`; evaluation and reporting take
`--root`. Use a separate `--root` for new work.

```sh
RUN_ROOT="$HOME/.graspkit/data/runs/my_grasp_policy_run"
SCENES="$HOME/.graspkit/data/scene_description/dex1b/v0.5.0/tabletop_mini.msgpack"
SCRIPTS=scripts/benchmark/grasp_policy

uv run "$SCRIPTS/train.py" --help
uv run "$SCRIPTS/package.py" --help
uv run "$SCRIPTS/evaluate.py" --help
uv run "$SCRIPTS/report.py" --help
```

Run long jobs on an approved compute host, with an explicitly assigned GPU.
`collect.py --stage label` replays the closed-loop protocol on training scenes in
any simulator: it labels GT grasps (per-scene files of the grasp optimizer) and
policy proposals from the initial state and from the step-75 feedback state, one
resumable HDF5 shard per scene batch (local or S3; shard with `--rank`/`--world-size`).
`--stage assemble` merges shard sets of several simulators and rounds into a bank
(`data.h5`, `replay.pt`, `scenes.msgpack`) that `train.py --bank` trains on.
`--sensor` accepts `depth` or `rgb` and defaults to `rgb`.

```sh
CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPTS/collect.py" --stage label --robot arx_x5 \
  --gt-dir "$GT_DIR" --checkpoint "$POLICY" --simulator maniskill --output "$RUN_ROOT/shards/arx_x5/maniskill"
uv run "$SCRIPTS/collect.py" --stage assemble --robot arx_x5 \
  --shards "$RUN_ROOT/shards/arx_x5/maniskill" --bank "$RUN_ROOT/banks/arx_x5"

CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPTS/train.py" --root "$RUN_ROOT" --bank "$RUN_ROOT/banks/arx_x5" \
  --robot arx_x5 --generator cvae --stage train \
  --steps 8000 --scorer-steps 10080 --objective scaled --grad-clip-norm 1
```

The custom `scaled` recipe uses per-phase action-standard-deviation MSE;
`--objective native` selects the native reconstruction objective. Gradient clipping
is explicit. `--steps 8000` executes 8,001 actor updates, including update zero.
The scorer's selected checkpoint step can differ from its 10,080-update run budget.
Recipes, selected steps and exclusions travel with each bundle. These settings do not silently replace ordinary GraspKit training defaults.

Prepare the dev split once with `evaluate.py --stage prepare`, supplying
`--banks` and `--policy-inventory`. Complete all 20 dev scenes before running a
condition's canonical 500 test scenes. The wrapper preserves seed
zero, 64 proposals, eight guidance steps and 300 control steps; closed control
captures fresh observations once at step 75 and replans the remaining 225 steps.
`--offset` and `--count` select exact
scene slices, including smaller recovery chunks.

```sh
POLICY="$RUN_ROOT/bundles/arx_x5/rgb/cvae/policy.pt"
CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPTS/evaluate.py" --stage run \
  --root "$RUN_ROOT" --scenes "$SCENES" --checkpoint "$POLICY" \
  --robot arx_x5 --sensor rgb --generator cvae \
  --simulator maniskill --loop closed --split dev --offset 0 --count 20

CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPTS/evaluate.py" --stage run \
  --root "$RUN_ROOT" --scenes "$SCENES" --checkpoint "$POLICY" \
  --robot arx_x5 --sensor rgb --generator cvae \
  --simulator maniskill --loop closed --split test --offset 0 --count 100
```

Repeat the test command at offsets 100, 200, 300 and 400. Isaac runs the same
`evaluate_policy` loop through `IsaacEnv` with `--simulator isaac`, stepping
`--num-envs` scenes together (every simulator); `--isaac-command` selects its separate simulator
Python process. Importing the pipeline or displaying CLI help does not start Isaac. `--simulator mujoco` runs
`MjNativeEnv`, a MuJoCo scene built from the same robot export, with the same cameras, controller
gains and success check. `complete.json` commits a chunk only after
the row and trajectory checks pass. Reporting counts actual marker trials and unique
scene IDs, including 8/4-scene recovery chunks.

The fresh-data retrain of 2026-10-15 lives in `DATA_ROOT/runs/retrain_20261015` (S3 copy
under the same name). Shards are grouped as `shards/<round>/<simulator>/<robot>`: round 0
labels GT grasps; later rounds label 64 proposals of earlier actors. ARX and ALOHA were
retrained the same way on 2026-10-16 (`runs/retrain_20261016`). Every v0.1 policy uses one
bank of all rounds and one recipe:

```sh
uv run "$SCRIPTS/collect.py" --stage assemble --robot arx_x5 \
  --shards "$RUN_ROOT"/shards/round*/*/arx_x5 --bank "$RUN_ROOT/banks/arx_x5"
CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPTS/train.py" --root "$RUN_ROOT/train" \
  --bundle-root "$RUN_ROOT/train/bundles" --bank "$RUN_ROOT/banks/arx_x5" \
  --robot arx_x5 --sensor depth --generator flow_transformer --stage train \
  --steps 16000 --batch-size 128 --grad-clip-norm 1
```

Results and the bars they are compared with are in the benchmark
[README](../../scripts/benchmark/grasp_policy/README.md).

The strict loader is `graspkit.models.checkpoint.load_policy`; checkpoints declare
normalization, reconstruction and grasp-evaluator feature options in `policy_config`. Four RGB
views remain paired with native camera metadata; joint order is recorded in
`config.json`.

`package.py` accepts `--policy-path`, `--actor-path`, `--bundle` and
`--validation` (the validation JSON).
`report.py --root "$RUN_ROOT"`
rebuilds the ledger under `<run>/native_report`; `--output` selects another
report directory.

Isaac evaluation enables GPU PhysX, matching RoboDojo's `overwrite_gpu_setting(1)`.
State tensors and image processing use CUDA, and Fabric propagates GPU transforms
before each observation render.
Seeds are per scene and phase, so outcomes do not depend on how scenes are batched.

Native robot materials and the textured table come from `graspkit.assets.isaac_appearance`
(HF dataset `data/isaac/appearance`), fetched on the first Isaac env.
