from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from jaxtyping import Float, Int
from robokit.xform.warp.torch_wrappers import inverse_tf_mat, rot_tl_to_tf_mat, transform_points

from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.grasp_robot import GraspRobot


class ContactExtractor:
    """Per-point graspness weights and best-action labels from a 30 deg palm cone (process_sim_data)."""

    def __init__(self, robot_config: RobotConfig, device: torch.device):
        self.robot_config = robot_config
        self.robot_helper = GraspRobot.from_config(robot_config, load_meshes=True, device=device)
        # Palm collision-mesh center of mass (link-local), anchors the hand x-axis in extract_contacts.
        palm_mesh = self.robot_helper.spec.get_link_mesh(
            robot_config.palm_link_name, mode="collision", return_mesh=True
        )
        self.palm_center_local = torch.as_tensor(palm_mesh.center_mass, dtype=torch.float, device=device)

    @torch.no_grad()
    def extract_contacts(
        self,
        object_points: Float[torch.Tensor, "num_points 3"],
        qpos: Float[torch.Tensor, "num_actions num_dofs"],
    ) -> Optional[
        Tuple[
            Float[torch.Tensor, "num_points"], Int[torch.Tensor, "num_points"], Int[torch.Tensor, "num_actions"]  # noqa: F821
        ]
    ]:
        """
        Extract the contact points and the action indices for each point; None if the result is degenerate.

        Returns:
            graspness_weights: The graspness weights for each point.
            action_indices: The best action indices for each point (-1: farther than 5 mm from every best point).
            best_point_indices: The best point indices for each action.
        """
        link_poses = self.robot_helper.forward_kinematics_torch(qpos)
        (thumb_tip_link_name,) = set(self.robot_config.thumb_link_names) & set(self.robot_config.fingertip_link_names)
        (middle_tip_link_name,) = set(self.robot_config.middle_link_names) & set(self.robot_config.fingertip_link_names)
        thumb_tip_link_idx = self.robot_helper.link_names.index(thumb_tip_link_name)
        middle_tip_link_idx = self.robot_helper.link_names.index(middle_tip_link_name)
        thumb_tip_pos = link_poses[..., thumb_tip_link_idx, :3, 3]
        middle_tip_pos = link_poses[..., middle_tip_link_idx, :3, 3]
        th_mf_center = (thumb_tip_pos + middle_tip_pos) / 2
        th_mf_center = th_mf_center.squeeze(-2)

        palm_link_idx = self.robot_helper.link_names.index(self.robot_config.palm_link_name)
        palm_link_pose = link_poses[..., palm_link_idx, :, :]
        palm_link_pose = palm_link_pose.squeeze(-3)

        palm_mesh_center = transform_points(
            repeat(self.palm_center_local, "d -> n 1 d", n=qpos.shape[0]), palm_link_pose
        ).squeeze(1)

        x_axis = th_mf_center - palm_mesh_center
        x_axis = F.normalize(x_axis, dim=-1, eps=1e-8)

        # project z [0, 0, 1] to the plane that orthogonal to axis
        z = torch.zeros_like(x_axis)
        z[..., 2] = 1
        z_axis = z - (z * x_axis).sum(dim=-1, keepdim=True) * x_axis
        z_axis = F.normalize(z_axis, dim=-1, eps=1e-8)

        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = F.normalize(y_axis, dim=-1, eps=1e-8)

        rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        poses = rot_tl_to_tf_mat(rot_mat, tl=palm_mesh_center)

        obj_pts_in_palm_coord = transform_points(object_points, inverse_tf_mat(poses))

        heights = obj_pts_in_palm_coord[..., 0]
        angles = torch.arccos(torch.abs(heights) / torch.norm(obj_pts_in_palm_coord, dim=-1))
        positive_mask = (heights > 0.0) * (angles < np.deg2rad(30.0))

        min_valid_height = heights.clone()
        min_valid_height[~positive_mask] = 100.0
        # min_height_points = min_valid_height.argmin(dim=-1)
        min_valid_height = min_valid_height.min(dim=-1, keepdim=True)[0]

        coef_angle = -180 / np.pi * np.log(2) / 10  # decay 50% at 10 degree
        coef_height = -np.log(2) / 0.015  # decay 50% at 1.5 cm
        angle_decay = coef_angle * angles  # [B,n]
        # decay w.r.t height from the point nearest in height, so palm-facing points get maximal graspness
        height_decay = coef_height * (heights - min_valid_height)  # [B,n]
        raw_weights = (angle_decay + height_decay).exp() * positive_mask  # .sum(dim=0) #[n]
        best_point_indices = raw_weights.argmax(dim=-1)

        best_points = object_points[best_point_indices]
        point_dists = (best_points[:, None] - object_points[None]).norm(dim=-1)
        point_dists, action_indices = torch.min(point_dists, dim=0)
        close_mask = point_dists < 0.005
        action_indices[~close_mask] = -1

        dist = (
            obj_pts_in_palm_coord
            - obj_pts_in_palm_coord[torch.arange(obj_pts_in_palm_coord.shape[0]), best_point_indices][:, None]
        ).norm(dim=-1)  # [num_actions, num_points]
        graspness_weights = ((10 ** (-dist * 150)) * positive_mask).sum(dim=0)
        graspness_weights = graspness_weights / (graspness_weights.mean() + 1e-6)

        if not torch.isfinite(graspness_weights).all() or graspness_weights.sum() < 1e-6 or action_indices.max() < 0:
            return None
        return graspness_weights, action_indices, best_point_indices
