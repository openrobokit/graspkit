# Real Deployment

Running a trained policy on hardware: capture a point cloud, predict a grasp, build a trajectory,
execute it, release, and park.

> These scripts move a real robot. Run them yourself, with the e-stop in reach — never from an
> automated agent.

## Prerequisites

- The rig's `xarm` extra, plus `sam2` for segmentation.
- A checkpoint whose robot matches the rig: deploy asserts that the actuated joint names of the
  policy's robot and the hardware robot are identical.

## Hardware presets

`configs/real_env.py` holds one `RealEnvConfig` per rig; `--real-env-config-name` selects it, and an
experiment can name a default through `real_env_config_name`.

| Preset | Arm | Hand | Camera |
| --- | --- | --- | --- |
| `xarm7_xhand_right` | xArm7 | XHand | RealSense |
| `xarm6_sharpa_right` | xArm6 | Sharpa | RealSense |
| `xarm7_ability_right` | xArm7 | Ability (channel reindex) | RealSense |

Each preset also carries the planning table (`table_size`, `table_xy`, `table_z`), the point-cloud
crop centre (`workspace_center`), the streaming rate (`control_hz`) and how many depth frames to
median-combine. Experiments need `--real-env-config-name`, a fixed-base trajectory and
`camera_frame="robot"`.

## Run it

```shell
uv run --extra sam2 --extra xarm scripts/deploy/deploy.py --env real \
  --exp-config-name xhand_right_dex1b_v0_5_mini_tabletop_fixed_eval \
  --ckpt-path <checkpoint> --real-env-config-name xarm7_xhand_right
```

Every stage waits for a button in the Viser page:

1. **Capture** — reset to the init pose, grab RGB (median of `depth_frames` depth frames).
2. **Segment** — click the object; SAM 2 turns the clicks into a mask, which becomes the cropped,
   subsampled point cloud in the robot base frame.
3. **Check Grasp** — the policy samples `--num-samples` grasps; the trajectory builder converts and
   plans them, and the top eight valid candidates are shown. For a floating-base policy the cloud is
   mean-centred and the centroid added back; `--pc-offset` nudges it. `--post-grasp-hand-close-scale`
   rescales the closing motion.
4. **Deploy** — the trajectory is streamed phase by phase at the rig's `control_hz`. The
   *Deploy Until Step* slider stops mid-trajectory so you can compare commanded and measured joint
   positions against the live cloud, then resume.
5. **Release and park** — the hand opens in place, then a collision-checked plan returns the arm to
   the init pose against a table raised by 5 cm. There is no reset afterwards.

`--loop` keeps cycling; without it, a capture with no valid grasp is an error.

## Calibration

The deploy path reads a base-frame camera extrinsic and a camera intrinsic from the shipped
calibration files (`assets/calib.py`, wired into the experiment's `renderer.camera_poses_path` and
`camera_intrinsic_path`).

Inspect the result with `uv run scripts/tools/visualize_camera_extrinsic.py --experiment-name ...`
(offline) or `uv run --extra xarm scripts/deploy/visualize_calib.py` (live).

The park plan raises the table by 5 cm because the optimizer otherwise rides the collision margin.
