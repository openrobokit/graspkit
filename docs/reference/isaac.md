# Isaac backend

Isaac follows the same layout as ManiSkill and MuJoCo:

- `src/graspkit/envs/isaac.py`: `IsaacEnv(BaseEnv)`, the graspkit side.
- `src/graspkit/utils/urdf_utils.py`: the backend-neutral sim URDF ManiSkill also uses.
- `EnvConfig.cameras`: the cameras, set per experiment (`configs/experiment/grasp_policy.py`).
- `src/graspkit/isaac/job.py`: exports the robot (that URDF, joint map, cameras, contact links, native materials)
  and the tabletop scenes to an HDF5 job.
- `src/graspkit/isaac/worker.py`: the Isaac process; starts Kit and serves `IsaacEnv` requests over a socket.
- `src/graspkit/isaac/dex_grasp.py`: the simulation the worker runs.
- `IsaacEnvConfig`, `EnvConfig.isaac_env_config` and `build_env`: configuration and backend selection.

Isaac Sim ships its own Python (`/isaac-sim/python.sh`) and its modules import only after
`AppLauncher` starts, so graspkit cannot import it: `IsaacEnv` starts the worker, writes each
reset's scenes to a job file and exchanges NumPy packets that carry dtype, shape and bytes.
Importing the adapter or selecting another backend does not import Isaac.

`IsaacEnv` implements `BaseEnv`, including the closed-loop policy primitives, so
`scripts/benchmark/grasp_policy/evaluate_policy.py` drives Isaac and ManiSkill with the same loop:

```python
from graspkit.configs.env_config import IsaacEnvConfig
from graspkit.envs import build_env

evaluator = replace(cfg.grasp_optim.evaluator, num_envs=8, isaac_env_config=IsaacEnvConfig())
env = build_env(evaluator, cfg.robot_config, "cuda:0", renderer_cfg=cfg.renderer)
env.reset(scenes, init_qpos=qpos, root_pose=root_pose)  # one env per scene
frame = env.observe()  # rgb (B, V, H, W, 3), xyz (B, N, 3) without the robot, camera_poses (B, V, 7)
env.step_qpos(targets)  # (B, D) pruned chain order
env.set_object_pose(env.get_object_pose() + offset)
info = env.get_info()  # success = lifted by target_lift_dist with at least two contact links touching
env.close()
```

The task (control rate, object mass, lift height) comes from `EnvConfig`, the
same as ManiSkill. The worker builds one GPU stage per reset with fixed-base robots, convex
object pieces and the table; physics runs at 100 Hz (`substeps=2`). Cameras are the robot's
benchmark cameras; a few camera rigs visit the envs in rounds, and frames are resampled to
the requested intrinsics. Robot pixels are removed by semantic segmentation, and points are
cropped and resampled per view like ManiSkill's renderer. Appearance follows SAPIEN: native
robot materials and the textured table from `IsaacEnvConfig.appearance`, vertex-colored
objects and a black background. `evaluate` replays trajectories in batches of `max_envs`.

`worker_command` defaults to `/isaac-sim/python.sh`.
Outside an installed Isaac runtime, configure a foreground container command that exposes the
source, robot assets and socket/cache directory at the same absolute paths. The library does
not choose GPUs. Worker logs, jobs and
the imported robot USD cache stay under `LOG_ROOT/isaac/`.
