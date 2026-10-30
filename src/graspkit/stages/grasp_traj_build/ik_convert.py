# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""Floating->fixed IK conversion via batched IK: a packed-tensor core function plus a `GraspKitData`-batch wrapper."""

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F
from robokit.utils.ee_coord_utils import ee_coord_conversion
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, inverse_tf_mat, rot_tl_to_tf_mat

from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.grasp_robot import GraspRobot, hand_coord_spec
from graspkit.scene.scene_description import SceneDescription

if TYPE_CHECKING:
    from graspkit.datasets.graspkit_data import GraspKitData


logger = logging.getLogger("graspkit")


def ik_convert_floating_to_fixed(
    scene_descriptions: List[SceneDescription],
    num_grasps_per_scene: torch.Tensor,
    pre_grasp_qpos: torch.Tensor,
    grasp_qpos: torch.Tensor,
    post_grasp_qpos: torch.Tensor,
    floating_robot_config: RobotConfig,
    fixed_robot: GraspRobot,
    arm_range_fraction: float = 0.30,
    target_xy: Optional[torch.Tensor] = None,
    ori_threshold_deg: float = 45.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """IK-convert packed floating-base grasps (pre/grasp/post) to fixed-base `[pos(3), euler_rxyz(3), hand_joints]`.

    - `arm_range_fraction`: half-width (fraction of range) around midpoint the arm IK must fit in; 1.0 disables.
    - `target_xy` (num_scenes, 2): overrides the orientation-filter target; None uses the scene's mesh position.
    - `ori_threshold_deg`: orientation-filter cone half-angle.
    - Returns `(pre_full, grasp_full, post_full, ik_success_mask)`; mask encodes orientation + IK + arm-range checks.
    """

    device = fixed_robot.device
    fixed_config = fixed_robot.config
    counts = num_grasps_per_scene.to(device)
    n = int(pre_grasp_qpos.shape[0])

    phases = torch.stack(
        [pre_grasp_qpos.to(device), grasp_qpos.to(device), post_grasp_qpos.to(device)], dim=1
    )  # (N, 3, dof)

    target_pose = rot_tl_to_tf_mat(
        rot_mat=euler_angles_to_matrix(phases[..., 3:6], axes="rxyz"), tl=phases[..., :3]
    )  # (N, 3, 4, 4)

    # Floating orientation is in the floating hand frame; rotate into the fixed frame before IK (no-op if coords match).
    if hand_coord_spec(floating_robot_config) != hand_coord_spec(fixed_config):
        coord_tf = ee_coord_conversion(
            hand_coord_spec(floating_robot_config), hand_coord_spec(fixed_config), return_tensors="pt"
        ).to(device)
        target_pose[..., :3, :3] = target_pose[..., :3, :3] @ coord_tf.T

    # Orientation filter: grasp-phase four-fingers direction should roughly face the target.
    if target_xy is None:
        target_xy = torch.tensor(
            [d.meshes[d.target_actor_id].position[:2] for d in scene_descriptions], device=device, dtype=torch.float32
        )
    base_xy = torch.tensor(fixed_config.workspace_origin[:2], device=device, dtype=torch.float32)
    base_to_tgt_xy = F.normalize(target_xy.to(device) - base_xy, dim=-1).repeat_interleave(counts, dim=0)
    four_fingers_in_hand = torch.tensor(hand_coord_spec(fixed_config).four_fingers_up_vec, device=device)
    four_fingers_xy = F.normalize((target_pose[:, 1, :3, :3] @ four_fingers_in_hand)[..., :2], dim=-1)
    ori_ok = (four_fingers_xy * base_to_tgt_xy).sum(dim=-1) > math.cos(math.radians(ori_threshold_deg))

    # IK: solve grasp first (init at rest), then pre/post from grasp; targets are relative to arm_base_link, not root.
    arm_base_pose = (
        fixed_robot.forward_kinematics_torch(fixed_robot.init_qpos[None, :])[
            0, fixed_robot.link_names.index(fixed_config.arm_base_link_name)
        ]
        if fixed_config.arm_base_link_name is not None
        else fixed_robot.root_pose_mat
    )
    target_in_armbase = inverse_tf_mat(arm_base_pose) @ target_pose  # (N, 3, 4, 4)
    init_arm = fixed_robot.init_qpos[fixed_robot.arm_joint_indices]
    arm_limits = fixed_robot.actuated_joint_limits_torch[fixed_robot.arm_joint_indices]
    ik = fixed_robot.inverse_kinematics
    grasp_jv, grasp_ok = ik(
        target_in_armbase[:, 1], init_qpos=init_arm, position_tolerance=0.005,
        orientation_tolerance=0.05, init_sample_range=0.3, use_score_rest=True,
    )  # fmt: skip
    # Joints with a range wider than a full turn (xarm7 wrist/base: +-2pi) admit +-2pi-equivalent solutions.
    turn = 2.0 * math.pi
    unwrapped = grasp_jv + torch.round((init_arm - grasp_jv) / turn) * turn
    grasp_jv = torch.where((unwrapped >= arm_limits[:, 0]) & (unwrapped <= arm_limits[:, 1]), unwrapped, grasp_jv)
    pre_jv, pre_ok = ik(
        target_in_armbase[:, 0], init_qpos=grasp_jv, position_tolerance=0.005,
        orientation_tolerance=0.05, init_sample_range=0.0, use_score_rest=True,
    )  # fmt: skip
    post_jv, post_ok = ik(
        target_in_armbase[:, 2], init_qpos=grasp_jv, position_tolerance=0.005,
        orientation_tolerance=0.05, init_sample_range=0.0, use_score_rest=True,
    )  # fmt: skip

    arm_jv = torch.stack([pre_jv, grasp_jv, post_jv], dim=1)  # (N, 3, arm_dof)
    joint_range = arm_limits[:, 1] - arm_limits[:, 0]
    joint_mid = (arm_limits[:, 1] + arm_limits[:, 0]) / 2
    # Locked joints (range==0) auto-pass -- IK returns ~0 plus FP noise that would fail an exact-equality band check.
    in_band = (arm_jv >= joint_mid - arm_range_fraction * joint_range) & (
        arm_jv <= joint_mid + arm_range_fraction * joint_range
    )
    arm_range_ok = (in_band | (joint_range == 0)).all(dim=-1).all(dim=-1)
    all_ik_ok = pre_ok & grasp_ok & post_ok
    ik_success_mask = ori_ok & all_ik_ok & arm_range_ok

    full_qpos = fixed_robot.init_qpos.expand(n, 3, -1).clone()
    full_qpos[:, :, fixed_robot.arm_joint_indices] = arm_jv
    full_qpos[:, :, fixed_robot.hand_joint_indices] = phases[..., 6:]

    logger.info(
        f"IK conversion: ori={int(ori_ok.sum())}, ik={int(all_ik_ok.sum())}, "
        f"range={int(arm_range_ok.sum())}, final={int(ik_success_mask.sum())} / {n}"
    )

    return full_qpos[:, 0], full_qpos[:, 1], full_qpos[:, 2], ik_success_mask


def ik_convert_floating_to_fixed_batch(
    batch: "GraspKitData",
    floating_robot_config: RobotConfig,
    fixed_robot_helper: GraspRobot,
    arm_range_fraction: float = 0.30,
    target_xy: Optional[torch.Tensor] = None,
    ori_threshold_deg: float = 45.0,
) -> "Tuple[GraspKitData, torch.Tensor]":
    from graspkit.datasets.graspkit_data import GraspKitData

    pre, grasp, post, ik_success_mask = ik_convert_floating_to_fixed(
        batch.scene_descriptions,
        batch.num_grasps_per_scene,
        batch.pre_grasp_qpos,
        batch.grasp_qpos,
        batch.post_grasp_qpos,
        floating_robot_config,
        fixed_robot_helper,
        arm_range_fraction,
        target_xy,
        ori_threshold_deg,
    )
    return GraspKitData(
        scene_ids=batch.scene_ids,
        scene_descriptions=batch.scene_descriptions,
        first_indices=batch.first_indices,
        pre_grasp_qpos=pre,
        grasp_qpos=grasp,
        post_grasp_qpos=post,
    ), ik_success_mask
