# Simulation

GraspKit uses two physics backends behind one interface: ManiSkill (GPU, batched, full rollouts) and
MuJoCo (CPU, grasp-pose scoring). A third path, the renderer, uses the ManiSkill scene without
stepping physics at all.

## One interface

`BaseEnv` fixes the contract: `reset`, `get_qpos`, `step_qpos`, `get_info`, `get_rgb`,
`get_pointcloud`, `evaluate`, `close`, with `evaluate` returning per-grasp success and, when asked,
`Rollout` records (`qpos`, `obj_pose`, `scene_id`, `success`, `grasp_index`).

`build_env` picks the implementation: `MjEnv` when `EnvConfig.mj_env_config` is set, otherwise
`MsEnv`; `RealEnv` is imported lazily so the sim path never needs the hardware extras. `MjEnv`
implements only `evaluate` — it scores grasps offline and raises on the interactive methods.
`RealEnv` is the mirror image: it drives hardware but has no ground truth, so `evaluate` raises.

## ManiSkill

The env is registered as `DexGrasp`. A scene description becomes actors grouped by mesh and scale,
built in sorted order because PhysX contact outcomes depend on insertion order, with mass forced to
`object_mass` and a merged `target` actor per scene. A table is added unless `floating_object` is
set, which also disables gravity on the object. Success is "strictly grasped and lifted past
`target_lift_dist`", or grasp-only for a floating object; grasping itself means at least two contact
links pushing with more than half a newton.

Floating hands can be driven two ways:

- **Virtual joints** — the agent's first six joints are PD-driven by the `floating_base` controller.
  This is the default for `*_floating_base` agent uids.
- **Static base** — when the ManiSkill agent is hand-only but the pipeline robot is floating, the
  six base DOFs are stripped from the qpos and applied as per-environment root poses instead. The
  base is pinned at the final grasp pose, which tests the grasp without an approach sweep or base PD
  shake.

Joint order differs between ManiSkill's observation order (`robot.active_joints`) and its action
order (`controller.joints`); `reindex_utils` converts in both directions, and picking the wrong one
silently scrambles a bimanual robot.

Agent uids follow the robot config name, with `_floating_base` appended for floating hands. The env
lists the 21 supported uids.

## MuJoCo

`MjDexGraspEnv` builds a model with MjSpec: the hand's MJCF, a table box that collides with the
object but never the hand, and the object as convex pieces added in a fixed order because MuJoCo
sums geom inertia in that order. Gravity is disabled; a downward wrench stands in for it. Density is
rescaled so the object weighs `obj_mass`.

Two protocols:

- **`fc`** — force closure in free space: pregrasp, grasp, squeeze, then hold against six
  axis-aligned forces scaled by `fc_force_scale`; failure is translation past `trans_thre` or
  rotation past `angle_thre`.
- **`tabletop`** — grip, apply the wrench, lift 20 cm (mocap root for floating hands, damped
  least-squares arm IK for fixed-base rigs), then require slip below `lift_slip_thre`, rotation
  below `angle_thre`, and live contact after the lift.

Both bail out immediately when the initial pose is already penetrating past `max_pene`. Beyond
success, the evaluator reports hand-object and self penetration, contact count, contact distance and
consistency. A `HandSpec` adapts a hand: its MJCF path, whether it is mocap-driven, finger prefixes,
the bodies that count as contact and fingertips, and — for fixed-base rigs — which body to lift and
which joints the IK may move.

MuJoCo is CPU-only and not thread-safe, so `MjEnv` runs a spawned pool of up to 48 workers, one
evaluator each. Spawn, not fork: forking a CUDA-initialized parent deadlocks.

## The renderer

`mani_skill/renderer.py` (`build_renderer`) makes its own `DexGrasp` env with `num_envs` environments and a point-cloud
observation mode, then never steps physics: it writes qpos and object poses, flushes the GPU
kinematics, and reads observations. Batching is over (scene, view) pairs. Segmentation ids select
what survives (`extract_segments` wins over `remove_segments`), and points are cropped either in the
camera frame or in the world frame offset by each scene's table position. `MsEnv.get_pointcloud`
replicates that crop exactly, so deployment clouds match the training distribution.

## Choosing a backend

| | ManiSkill | MuJoCo |
| --- | --- | --- |
| Hardware | GPU, batched | CPU, process pool |
| Scores | Full approach → grasp → lift rollouts, with perturbation forces | A grasp pose: force closure or a lift |
| Gives | Success, observations, RGB and point clouds | Success plus penetration and contact metrics |
| Robots | The 21 agents with ManiSkill uids | Any hand with an MJCF `HandSpec` |

Hands with no ManiSkill agent (menagerie Shadow, the `_mj` xArm+XHand and Vega+Sharpa rigs) can only
be validated in MuJoCo, which is why several experiments use it as their in-loop grasp filter.
