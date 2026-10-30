"""DexJoCo Panda and Allegro model, frames, and native MuJoCo position controller."""

from dataclasses import replace
from pathlib import Path

import mujoco
from robokit.assets.robots.assembly import franka_panda_allegro
from robokit.assets.robots.end_effectors import allegro_hand

from graspkit.configs.env_config import HandSpec
from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.hand_mount import mjcf_with_hand_mount

_ARM_PD = (600.0, 10.0)
_KINEMATIC_XML = Path(
    mjcf_with_hand_mount(
        franka_panda_allegro.MJCF_PATH_RIGHT,
        hand_link="allegro_palm_right",
        floating_offset=(0.035, 0.0, 0.014),
    )
)
PANDA_ALLEGRO_MJ_XML = str(_KINEMATIC_XML.with_stem(f"{_KINEMATIC_XML.stem}_pd_{_ARM_PD[0]:g}_{_ARM_PD[1]:g}"))
if not Path(PANDA_ALLEGRO_MJ_XML).exists():
    _spec = mujoco.MjSpec.from_file(str(_KINEMATIC_XML))
    for _actuator in _spec.actuators[:7]:
        _actuator.forcerange = _actuator.ctrlrange.copy()
        _actuator.forcelimited = True
        _actuator.ctrlrange = (0.0, 0.0)
        _actuator.set_to_position(kp=_ARM_PD[0], kv=_ARM_PD[1], inheritrange=True)
    _spec.compile()
    _spec.to_file(PANDA_ALLEGRO_MJ_XML)

panda_allegro_right_mj = RobotConfig(
    name="panda_allegro_right_mj",
    urdf_or_mjcf_path=PANDA_ALLEGRO_MJ_XML,
    collision_spheres_path=str(franka_panda_allegro.MJ_COLLISION_SPHERE_PATH_RIGHT),
    base_type="fixed",
    keep_world_link=True,
    # Keep the vendor world root; its pose also places the MJCF during evaluation.
    init_position=(0.8, 0.05, -0.924),
    workspace_origin=(0.0, -0.35, 0.0),
    arm_base_link_name="link0_right",
    hand_base_link_name="allegro_palm_right_floating_root",
    ee_coord_spec=allegro_hand.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name="allegro_palm_right",  # the vendor palm, oriented like robokit's allegro base_link,
    init_qpos=(0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0)
    + (0.0, 0.44, 0.33, 0.52, 0.0, 0.19, 0.73, 0.15, 0.0, 0.29, 0.56, 0.38, 1.2, 0.0, 0.0, 0.0),
    palm_link_name="allegro_palm_right",
    thumb_link_names=["th_proximal_right", "th_medial_right", "th_distal_right", "th_tip_right"],
    middle_link_names=["mf_proximal_right", "mf_medial_right", "mf_distal_right", "mf_tip_right"],
    fingertip_link_names=["ff_tip_right", "mf_tip_right", "rf_tip_right", "th_tip_right"],
    contact_link_names=tuple(
        f"{finger}_{segment}_right" for finger in ("ff", "mf", "rf", "th") for segment in ("distal", "tip")
    ),
)

panda_allegro_hand_mj = replace(
    panda_allegro_right_mj,
    name="panda_allegro_hand_mj",
    urdf_or_mjcf_path=mjcf_with_hand_mount(PANDA_ALLEGRO_MJ_XML, floating_root="allegro_palm_right_floating_root"),
    base_type="floating",
    keep_world_link=False,
    init_position=(0.0, 0.0, 0.0),
    workspace_origin=(0.0, 0.0, 0.0),
    arm_base_link_name=None,
    init_qpos=panda_allegro_right_mj.init_qpos[7:],
)

PANDA_ALLEGRO = HandSpec(
    xml_path=PANDA_ALLEGRO_MJ_XML,
    mocap=False,
    finger_prefix=[f"{finger}_" for finger in ("ff", "mf", "rf", "th")],
    valid_body_name=["allegro_palm_right"]
    + [
        f"{finger}_{segment}_right"
        for finger in ("ff", "mf", "rf", "th")
        for segment in ("proximal", "medial", "distal", "tip")
    ],
    tip_body_name=[f"{finger}_tip_right" for finger in ("ff", "mf", "rf", "th")],
    lift_arm_body="allegro_palm_right",
    arm_qpos_indices=tuple(range(7)),
)
