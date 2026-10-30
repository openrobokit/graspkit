"""Every robot config's hand frame (`hand_coord_spec`, derived from robokit's `ee_coord_spec`) against its geometry.

Hands: at the open (zero) pose the fingertips lie along four_fingers_up, and the thumb sits on -palm_right for a right
hand (+palm_right for a left one). Grippers: the jaws lie along palm_forward, the axis the init sampler aims at the
object, and sampled inits put the jaws between the hand and the object.
"""

import numpy as np
import pytest
import torch
import trimesh
from robokit.utils.ee_coord_utils import GripperCoordinateSpec

from graspkit.configs.robot import _ROBOT_MODULES, build_robot_config
from graspkit.robot.grasp_robot import GraspRobot, hand_coord_spec
from graspkit.stages.grasp_optim import (
    BoxDescription,
    GraspOptimConfig,
    MeshDescription,
    SceneDescription,
    SceneGeom,
    build_sampler,
)

# franka_gripper and xarm7_sharpa_right name no fingertips; shadow_right's URDF does not parse.
SKIP = {"franka_gripper", "xarm7_sharpa_right", "shadow_right"}
COMPOSED = ["yam_yam_gripper", "franka_panda_allegro_right", "xarm7_sharpa_right", "ur5e_robotiq_2f85"]


@pytest.mark.hf_assets
class TestRobotCoordSpecs:
    @pytest.mark.parametrize("name", [n for n in _ROBOT_MODULES if n not in SKIP] + [f"+{n}" for n in COMPOSED])
    def test_hand_frame_matches_geometry(self, name: str):
        from graspkit.utils.assembly_utils import assembled_robot_config  # imports robot assets

        cfg = assembled_robot_config(name[1:]) if name.startswith("+") else build_robot_config(name)
        spec = hand_coord_spec(cfg)
        robot = GraspRobot.from_config(cfg, prune=False)
        limits = robot.actuated_joint_limits_torch
        q = torch.zeros(robot.num_actuated_joints).clamp(limits[:, 0], limits[:, 1])
        poses = robot.forward_kinematics_torch(torch.cat([torch.zeros(6 if cfg.base_type == "floating" else 0), q]))
        to_hand = torch.linalg.inv(poses[robot.link_names.index(cfg.hand_base_link_name or robot.link_names[0])])
        tips = {
            link: (to_hand @ poses[robot.link_names.index(link)])[:3, 3].numpy() for link in cfg.fingertip_link_names
        }
        center = np.mean(list(tips.values()), axis=0)
        if isinstance(cfg.ee_coord_spec, GripperCoordinateSpec):
            assert center @ spec.palm_forward_vec > 0.7 * np.linalg.norm(center), (
                f"jaws are not along palm_forward {spec}"
            )
            return
        assert center @ spec.four_fingers_up_vec > 0.7 * np.linalg.norm(center), (
            f"fingers are not along four_fingers_up {spec}"
        )
        thumb = [link for link in tips if link in cfg.thumb_link_names or "thumb" in link]  # some tips are pad links
        others = np.mean([p for link, p in tips.items() if link not in thumb], axis=0)
        side = 1.0 if "left" in cfg.name else -1.0
        assert np.sign((tips[thumb[0]] - others) @ spec.palm_right_vec) == side, f"thumb on the wrong side of {spec}"

    @pytest.mark.parametrize(
        "name", ["arx_x5_gripper", "agilex_aloha_gripper", "yam_gripper", "robotiq_2f85", "dclaw_right"]
    )
    def test_sampled_inits_face_the_object(self, tmp_path, name: str):
        cube = str(tmp_path / "cube.obj")
        trimesh.creation.box((0.06, 0.06, 0.06)).export(cube)
        scene = SceneDescription(
            scene_id="s0",
            target_actor_id="object",
            boxes={"table": BoxDescription("table", (0.5, 0.5, 0.05), (0.0, 0.0, -0.5))},
            meshes={"object": MeshDescription("object", cube, (1.0, 1.0, 1.0))},
        )
        cfg = build_robot_config(name)
        robot = GraspRobot.from_config(cfg, device="cpu")
        sampler = build_sampler(GraspOptimConfig(), cfg, robot, "cpu", sampler_type="floating_base")
        inits = sampler.sample(SceneGeom([scene], device="cpu"), 32)[0]
        limits = robot.actuated_joint_limits_torch
        assert ((inits[:, 6:] >= limits[:, 0]) & (inits[:, 6:] <= limits[:, 1])).all(), (
            "the jitter left a joint's range"
        )
        poses = robot.forward_kinematics_torch(inits)
        jaws = torch.stack([poses[:, robot.link_names.index(link), :3, 3] for link in cfg.fingertip_link_names]).mean(0)
        to_jaws = torch.nn.functional.normalize(jaws - inits[:, :3], dim=-1)
        to_object = torch.nn.functional.normalize(-inits[:, :3], dim=-1)  # the cube sits at the origin
        assert (to_jaws * to_object).sum(-1).mean() > 0.8
