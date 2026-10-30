# Sim Deployment

Three ways to run a trained policy in simulation: inside training, as a reproducible benchmark, or
interactively in the deploy UI.

## During training

Set `TrainerConfig.evaluator` to an `EnvConfig` and the trainer evaluates every `eval_freq`
iterations on rank 0: it samples actions, builds trajectories with `GraspTrajBuilder`, rolls them
out, and reports a success rate plus a report-only penetration pass rate
(`penetration <= eval_pen_threshold`, 5 mm by default).

```shell
# evaluate an existing checkpoint, no training
uv run scripts/stages/train/train.py <experiment> --trainer.eval-only --ckpt-path <ckpt>
```

Results land in `logs/train/<experiment>/eval_success.jsonl`, TensorBoard (`eval/success_rate`,
`eval/pen_pass_rate`) and `checkpoints/best_eval_success_rate.pt`. Eval-only runs also write
per-scene results to `logs/eval/<experiment>.json`.

Knobs on `EnvConfig`: `num_envs` (2048), `num_seeds` (a grasp must survive every seed),
`enable_perturbation` with `perturbation_magnitude` / `perturbation_step_range` /
`num_perturbations`, the task (`control_freq`, `object_mass`, `target_lift_dist`, ...), and
`mj_env_config` / `isaac_env_config` to pick a backend other than ManiSkill. `TrainerConfig`
adds `eval_num_envs`, `max_eval_batches` and `eval_pen_threshold`.

ManiSkill counts a tabletop grasp as a success when the hand is strictly grasping and the object
has risen by `target_lift_dist`; with `floating_object` it is grasp-only.

## As a benchmark

The default checkpoint mode of `scripts/benchmark/grasp_policy/evaluate_policy.py` makes every prediction before
any physics, asserts the eval scenes are disjoint from the training and validation IDs, and records
actions, trajectories, and per-scene results.

```shell
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=<free gpu> uv run scripts/benchmark/grasp_policy/evaluate_policy.py \
  --checkpoint <ckpt> --output <report.json> \
  --evaluation-experiment <experiment> --observations-path <processed.h5> \
  --require-native --require-object-disjoint
```

The checkpoint must carry `experiment`, `policy_config`, `agent` and `training` keys; plain trainer
checkpoints hold only `agent`, so use the packaging scripts under `scripts/stages/train/`. The
report gives `raw` (sim success), `penetration` (deepest hand-sphere overlap into the object at the
grasp pose, in metres) and `joint` = `raw & penetration <= 0.005`. That 5 mm threshold is fixed in
the script, independent of `eval_pen_threshold`.

The same runner supports tabletop observations with `--mode pc_open_loop`, `pc_closed_loop`, or
`rgb_closed_loop`. Choose `--robot` from `arx_x5`, `panda_allegro`, or `agilex_aloha`, and
`--arm-mode` from `floating_ik` or `arm_aware`. Each scene gets one selected grasp execution; tabletop results
are written to a `.jsonl` output.

## Interactively

```shell
uv run --extra sam2 scripts/deploy/deploy.py --env sim \
  --exp-config-name <experiment> --ckpt-path <ckpt>
```

The sim path runs one environment, takes scenes from the experiment's msgpack split, and drives the
same staged Viser UI the real robot uses — Capture, Segment, Check Grasp, Deploy. `--auto` (sim
only) takes the segmentation prompt from the simulator's ground-truth mask and prints the SAM2-vs-GT
mask IoU; `--scene-id` and `--num-scenes` choose scenes, `--deploy-batch-size` replays several
grasps at once.

## MuJoCo instead of ManiSkill

Setting `EnvConfig.mj_env_config` selects MuJoCo, which is the only backend for hands with no
ManiSkill agent (`shadow_right_mj`, `xarm7_xhand_right_mj`, `vega_sharpa_*_mj`). It runs on CPU in a
spawned pool of up to 48 workers and evaluates a grasp pose rather than a trajectory:

- `setting="fc"` — free-space force closure: grip, then hold against six axis-aligned forces; fails
  if the object translates past `trans_thre` or rotates past `angle_thre`.
- `setting="tabletop"` — grip, apply a downward wrench (gravity is disabled), lift 20 cm, and
  require slip below `lift_slip_thre`, rotation below `angle_thre`, and contact after the lift.

Use `timestep=0.001` for actuated arm+hand models; the 0.004 default flings the object.

## Pitfalls

- `MsEnv` warns when `num_envs` and the sample count are not multiples of each other: the mismatch
  forces a reconfiguration per batch, which dominates the runtime.
- Evaluating far fewer grasps than `num_envs` is wasteful; the env warns about that too.
- Reconfiguration only happens when the batch's scene set changes, so sort work by scene.
- Seeding is explicit: `evaluate_policy.py` asserts `PYTHONHASHSEED=0`; `seed_everything` does not
  enable deterministic kernels unless asked.
