# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""FloatingBaseIKScorer: ranks floating-hand grasps by how reachable a fixed-base arm finds them."""

import logging
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from robokit.utils.ee_coord_utils import ee_coord_conversion
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, inverse_tf_mat, rot_tl_to_tf_mat

from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.grasp_robot import GraspRobot, hand_coord_spec
from graspkit.scene.scene_description import SceneDescription

logger = logging.getLogger("graspkit")


class FloatingBaseIKScorer:
    """Fixed-base IK reachability scoring for floating-base grasps."""

    def __init__(
        self,
        floating_robot_config: RobotConfig,
        fixed_robot_config: RobotConfig,
        scene_descriptions: List[SceneDescription],
        num_samples_per_scene: int,
        device: torch.device,
    ):
        self.device = device
        self.num_samples_per_scene = num_samples_per_scene
        self.n_scene = len(scene_descriptions)
        self.fixed_robot = GraspRobot.from_config(fixed_robot_config, device=device)

        # Rotate floating grasp orientation into the fixed hand frame before IK; None (no-op) when coord specs match.
        self.coord_tf = None
        self.coord_tf_mat = None
        if hand_coord_spec(floating_robot_config) != hand_coord_spec(fixed_robot_config):
            self.coord_tf = ee_coord_conversion(
                hand_coord_spec(floating_robot_config), hand_coord_spec(fixed_robot_config), return_tensors="pt"
            ).to(device)
            self.coord_tf_mat = rot_tl_to_tf_mat(rot_mat=self.coord_tf)

        self.init_arm_qpos = self.fixed_robot.init_qpos[self.fixed_robot.arm_joint_indices]
        # IK targets are relative to arm_base_link, not the root link (e.g. G1's torso_link is offset).
        arm_base_pose = self.fixed_robot.forward_kinematics_torch(self.fixed_robot.init_qpos[None, :])[
            0, self.fixed_robot.link_names.index(fixed_robot_config.arm_base_link_name)
        ]
        self.arm_base_pos = arm_base_pose[:3, 3]
        self.world_to_armbase = inverse_tf_mat(arm_base_pose)
        arm_limits = self.fixed_robot.actuated_joint_limits_torch[self.fixed_robot.arm_joint_indices]
        self.arm_joint_range = arm_limits[:, 1] - arm_limits[:, 0]

        tabletop_pos = torch.tensor(
            [d.meshes[d.target_actor_id].position for d in scene_descriptions], device=device, dtype=torch.float32
        )
        base_xy = torch.tensor(fixed_robot_config.workspace_origin[:2], device=device, dtype=torch.float32)
        self.base_to_tgt_xy = F.normalize(tabletop_pos[:, :2] - base_xy, dim=-1)  # (n_scene, 2)
        self.four_fingers_in_hand = torch.tensor(hand_coord_spec(fixed_robot_config).four_fingers_up_vec, device=device)

    def compute_workspace_mask(
        self, variables: Float[torch.Tensor, "n dof"], scene_idx: int, workspace_radius: float = 0.8
    ) -> Bool[torch.Tensor, "..."]:
        """Keep init poses near the arm base whose four-fingers axis faces the target; `variables` = `[tl,rot,hand]`."""
        dist_ok = torch.norm(variables[..., :3] - self.arm_base_pos, dim=-1) < workspace_radius
        rot = euler_angles_to_matrix(variables[..., 3:6], axes="rxyz")
        if self.coord_tf is not None:
            rot = rot @ self.coord_tf.T
        four_fingers_xy = F.normalize((rot @ self.four_fingers_in_hand)[..., :2], dim=-1)
        ori_ok = (four_fingers_xy * self.base_to_tgt_xy[scene_idx]).sum(dim=-1) > math.cos(math.pi / 4)
        return dist_ok & ori_ok

    def filter_reachable(
        self, init_variables_oversampled: Float[torch.Tensor, "n_scene oversample dof"]
    ) -> Float[torch.Tensor, "n_scene num_samples dof"]:
        """Per scene, keep poses passing `compute_workspace_mask`, then truncate/pad to `num_samples_per_scene`."""
        num_samples = self.num_samples_per_scene
        oversample = init_variables_oversampled.shape[1]
        filtered_scenes = []
        for scene_idx in range(self.n_scene):
            scene_vars = init_variables_oversampled[scene_idx]
            passing = scene_vars[self.compute_workspace_mask(scene_vars, scene_idx)]
            if passing.shape[0] >= num_samples:
                filtered_scenes.append(passing[:num_samples])
            else:
                logger.warning(
                    f"Scene {scene_idx}: only {passing.shape[0]}/{oversample} pass workspace filter, "
                    f"padding to {num_samples}"
                )
                if passing.shape[0] > 0:
                    pad_indices = torch.randint(
                        0, passing.shape[0], (num_samples - passing.shape[0],), device=self.device
                    )
                    filtered_scenes.append(torch.cat([passing, passing[pad_indices]], dim=0))
                else:
                    perm = torch.randperm(oversample, device=self.device)[:num_samples]
                    filtered_scenes.append(scene_vars[perm])
        return torch.stack(filtered_scenes, dim=0)

    def select_reachable_seeds(
        self,
        all_q: Float[torch.Tensor, "n_total dof"],
        all_base_mat: Float[torch.Tensor, "n_total 4 4"],
        all_energy: Float[torch.Tensor, "..."],
    ) -> Tuple[Float[torch.Tensor, "n_scene num_samples 4 4"], Float[torch.Tensor, "n_scene num_samples dof"]]:
        """Soft-rank seeds by energy + `compute_penalty`; keep the lowest-cost `num_samples_per_scene` per scene."""
        num_samples = self.num_samples_per_scene
        seeds_per_scene = all_q.shape[0] // self.n_scene
        ik_penalty = self.compute_penalty(all_base_mat)
        scene_q = all_q.reshape(self.n_scene, seeds_per_scene, -1)
        scene_base = all_base_mat.reshape(self.n_scene, seeds_per_scene, 4, 4)
        scene_sort_key = (all_energy + ik_penalty).reshape(self.n_scene, seeds_per_scene)
        scene_penalty = ik_penalty.reshape(self.n_scene, seeds_per_scene)

        selected_q = []
        selected_base = []
        for s in range(self.n_scene):
            _, top_idx = torch.topk(scene_sort_key[s], k=num_samples, largest=False)
            selected_q.append(scene_q[s][top_idx])
            selected_base.append(scene_base[s][top_idx])
            n_passing = int((scene_penalty[s] < 1.0).sum())
            logger.info(f"Scene {s}: {n_passing}/{seeds_per_scene} seeds reachable (IK)")
        return torch.stack(selected_base), torch.stack(selected_q)

    def compute_penalty(self, base_mat: Float[torch.Tensor, "n 4 4"]) -> Float[torch.Tensor, "..."]:
        """Graded reachability cost for flattened (scene-major) grasp base poses in world frame."""
        target_pose = base_mat.clone()
        if self.coord_tf_mat is not None:
            target_pose = target_pose @ inverse_tf_mat(self.coord_tf_mat)
        target_in_armbase = self.world_to_armbase @ target_pose
        batch_size = target_in_armbase.shape[0]

        arm_q, ik_ok = self.fixed_robot.inverse_kinematics(
            target_in_armbase,
            init_qpos=self.init_arm_qpos.unsqueeze(0).expand(batch_size, -1),
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            init_sample_range=0.3,
        )
        arm_in_range = ((arm_q >= -0.30 * self.arm_joint_range) & (arm_q <= 0.30 * self.arm_joint_range)).all(dim=-1)

        four_fingers_xy = F.normalize((target_pose[:, :3, :3] @ self.four_fingers_in_hand)[:, :2], dim=-1)
        candidates_per_grasp = batch_size // (self.n_scene * self.num_samples_per_scene)
        scene_idx = torch.arange(batch_size, device=self.device) // (candidates_per_grasp * self.num_samples_per_scene)
        ori_ok = (four_fingers_xy * self.base_to_tgt_xy[scene_idx]).sum(dim=-1) > math.cos(math.pi / 4)

        return torch.where(~ik_ok, 1e6, torch.where(~ori_ok, 1e5, torch.where(~arm_in_range, 1e4, 0.0)))
