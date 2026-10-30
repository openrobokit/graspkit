# Robot and Hand Catalog

Every robot is a `RobotConfig` instance in `configs/robot/`, registered by name in that package's
`__init__.py`. `build_robot_config(name)` imports only that module, so only its assets download.

## Hands and rigs

| Name | Base | Rig | Format | DOF |
| --- | --- | --- | --- | --- |
| `ability_right` | floating | hand | URDF | 6 |
| `allegro_right` | floating | hand | URDF | 16 |
| `barrett_right` | floating | hand | URDF | 8 |
| `dclaw_right` | floating | 3-finger gripper | URDF | 9 |
| `inspire_right` | floating | hand | URDF | 12 |
| `leap_right` | floating | hand | URDF | 16 |
| `mano_right` | floating | hand | URDF | 16 |
| `orca_right` | floating | hand | URDF | 17 |
| `paxini_right` | floating | hand | URDF | 16 |
| `schunk_right` | floating | hand | URDF | 9 |
| `shadow_right`, `shadow_right_mj` | floating | hand | MJCF | 22 |
| `sharpa_right`, `sharpa_left` | floating | hand | URDF | 22 |
| `xhand_right`, `xhand_right_mj` | floating | hand | URDF / MJCF | 12 |
| `robotiq_2f85`, `yam_gripper` | floating | gripper | URDF | 1–2 driven |
| `franka_gripper` | fixed | arm + gripper | URDF | 8 |
| `xarm6_sharpa_right` | fixed | arm + hand | URDF | 6 + 22 |
| `xarm7_ability_right` | fixed | arm + hand | URDF | 7 + 6 |
| `xarm7_mano_right` | fixed | arm + hand | URDF | 7 + 16 |
| `xarm7_sharpa_right` | fixed | arm + hand | URDF | 7 + 22 |
| `xarm7_shadow_right` | fixed | arm + hand | MJCF | 7 + 22 |
| `xarm7_xhand_right`, `xarm7_xhand_right_mj` | fixed | arm + hand | URDF / MJCF | 7 + 12 |
| `g1_inspire_right`, `g1_inspire_right_wbc` | fixed | humanoid + hand | URDF | 41 / 33 |
| `g1_mano_right` | fixed | humanoid + hand | URDF | 45 |
| `vega_sharpa_right`, `vega_sharpa_left` | fixed | humanoid + hand | URDF | full body |
| `vega_sharpa_right_mj`, `vega_sharpa_left_mj` | fixed | humanoid + hand | MJCF | 63 |

Variants suffixed `_side_init` and `_centered` differ only in initial pose or camera framing;
`_mj` variants are the MJCF models used by the MuJoCo evaluator. Not every registered robot has a
pipeline: experiments exist for ability, inspire, mano, paxini, sharpa, xhand, the xArm rigs, the G1
rigs, Vega and the MuJoCo Shadow.

## What a config carries

Paths to the URDF or MJCF, collision spheres, self-collision ignore list, contact points and
contact-avoid points; the base type and initial pose; the link names that name the palm, fingertips,
thumb and middle finger; and robokit's `EE_COORD_SPEC` as `ee_coord_spec`, with the link it is expressed in
as `ee_coord_link_name`. `grasp_robot.hand_coord_spec` turns it into palm-forward and fingers-up axes in the hand
frame (a gripper's jaws are its palm). Those axes drive the hand-axis cone filter and the init sampler, so a wrong
spec shows up as grasps approaching from the wrong side; `tests/test_robot_coord_specs.py` checks every config's
axes against its fingertips and its sampled inits.

Contacts come either from a contact-points file or from a `contact_candidates_config` that samples
per link; `scripts/tools/create_contact_points.py` authors the former.

## ManiSkill agents

Simulation needs a ManiSkill agent per robot, in `mani_skill/agents/`. The uid is the robot config
name, with `_floating_base` appended for floating hands — `xhand_right_floating_base`,
`vega_sharpa_right`. Agents load sim-specific URDFs (floating-base, mimic-free variants) rather than
the config's own path, and compose a controller: the six virtual base joints plus a position
controller for the fingers. A robot with no hand-written agent (dclaw, orca, paxini, the grippers,
every assembled rig) gets one from `mani_skill/agents/config_agent.py`: `register_config_agent`
builds it from the robot config, writing a mimic-free sim URDF (plus the six virtual base joints for
a floating hand) under `~/.graspkit/data/sim_urdf/<revision>`. The `_mj` rigs are MuJoCo-only.

## Adding a hand

1. Write `configs/robot/<hand>.py`: one `RobotConfig` pointing at the robokit asset symbols.
2. Register it in `configs/robot/__init__.py`.
3. Add a ManiSkill agent in `mani_skill/agents/<hand>.py` and import it in `mani_skill/__init__.py`.
4. Author contact points if the hand needs them.
5. Optionally add it to `EES` in robokit's `assets/robots/assembly/composed.py` so it can be mounted on any arm,
   and to the gallery's hand list.
6. Add an experiment config when you want a pipeline for it.

`configs/robot/schunk.py` plus `mani_skill/agents/schunk.py` is the smallest complete example.

## Assembled rigs

robokit's `assets/robots/assembly/composed.py` assembles any of seven bases (YAM, xArm7, xArm6, Franka, UR5e,
G1, Vega) with any of fourteen end effectors, merging collision spheres and computing the mount transform from
the two mount specs, on demand under `~/.robokit/assembly/<revision>`. `utils/assembly_utils.py` adds each
base's scene placement and rest pose; its configs are what the gallery's arm dropdown offers. A base+ee name is always assembled, even where a
hand-written config shares it. `uv run python -m graspkit.gallery.grasp_optim --all --gpus ...` optimizes
every gallery hand floating and jointly on every base, sim-evaluates all of them, and prints the success table.
