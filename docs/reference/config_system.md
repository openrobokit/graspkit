# Config System

All configuration is plain dataclasses in `src/graspkit/configs/`, parsed with
[tyro](https://brentyi.github.io/tyro/) at the CLI. There are no YAML files: experiments are
hand-written Python instances, variants are derived with `dataclasses.replace`, and every field is
overridable from the command line. Field docstrings in the source are authoritative; this page
covers the structure and entry points.

## Layout

Class definitions live in `*_config.py`; instances live apart from them.

| Path (`src/graspkit/configs/`) | Contents |
|---|---|
| `experiment_config.py` | `ExperimentConfig` (top level) and all stage sub-configs |
| `experiment/` | Per-robot experiment instances + name registry (`build_experiment_config`, `cli_experiment`) |
| `robot_config.py` | `RobotConfig` |
| `robot/` | Per-robot `RobotConfig` instances + name registry (`build_robot_config`) |
| `grasp_policy_config.py` | `GraspPolicyConfig` |
| `grasp_policy.py` | Named `GraspPolicyConfig` presets (`GRASP_POLICY_CONFIGS`) |
| `env_config.py` | Backend configs: `MsEnvConfig` (ManiSkill), `MjEvalConfig` + `HandSpec` (MuJoCo), `IsaacEnvConfig`; the task is `EnvConfig` |
| `mj_hand.py` | MuJoCo `HandSpec` instances (`SHADOW_HAND`, `XHAND`, ...) |
| `real_env_config.py` | `RealEnvConfig` |
| `real_env.py` | `RealEnvConfig` hardware presets for `RealEnv` (`REAL_ENV_CONFIGS`; needs the `xarm` extra) |

## ExperimentConfig

`ExperimentConfig` (`experiment_config.py`) is the single config for the whole pipeline
(`launch.py` and every stage script). Sub-configs hold hand-independent algorithm/runtime params;
all paths, robot configs, and pipeline control live at the top level.

| Field | Type | Role |
|---|---|---|
| `name` | str | Experiment name (required) |
| `robot_config` | `RobotConfig` | Robot that grasp optimization and training target (the floating hand in the standard pipeline) |
| `traj_builder` | `GraspTrajBuilderConfig` | Floating/fixed robot pair, `convert_to_fixed` decision, grasp filters, and the builder's `MotionPlannerConfig` |
| `grasp_optim` | [`GraspOptimConfig`](api/grasp_optim.md) | Optimizer backend, energy weights, init sampler, in-loop `evaluator`: see [grasp_optim tutorial](../guide/grasp_optim.md) |
| `renderer` | `RendererConfig` | Point-cloud rendering: env, cameras (calibrated extrinsic/intrinsic paths), crop/downsample |
| `trainer` | [`TrainerConfig`](api/training.md) | Training loop: batch size, iters, eval `EnvConfig`, wandb |
| `dataset` | [`DatasetConfig`](api/training.md) | Dataset loading (num actions/points, graspness) |
| `agent` | [`GraspPolicyConfig`](api/policy.md) | Model architecture/loss preset: see [training tutorial](../guide/training.md) |
| `scene_description_path`, `grasp_output_dir`, `rendered_data_path`, `processed_data_path`, ... | str | All experiment-specific paths (usually under `DATA_ROOT`) |
| `steps`, `restart_from`, `append`, `gpus`, `num_gpus`, `seed` | - | Pipeline control for `launch.py` |
