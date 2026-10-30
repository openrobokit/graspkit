# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""Grasp candidate filters: thresholds in `__init__`, `__call__` -> bool mask; filters needing `qpos` assert on it."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from robokit.terms.dense.self_collision_task import compute_active_collision_pairs
from robokit.utils.ee_coord_utils import AXIS_TO_VEC
from torch.types import Device

from graspkit.configs.robot_config import HandAxis
from graspkit.robot.grasp_robot import GraspRobot, hand_coord_spec
from graspkit.scene.scene_description import SceneDescription
from graspkit.scene.scene_geom import SceneGeom


@dataclass
class Candidates:
    """One stage's candidates; pass `clst_pts` when known, as re-deriving from `tl` can differ on non-convex objects."""

    tl: Float[torch.Tensor, "n_scene n_cand 3"]
    rot: Float[torch.Tensor, "n_scene n_cand 3 3"]
    robot: GraspRobot
    scene_geom: SceneGeom
    qpos: Optional[Float[torch.Tensor, "n_scene n_cand dof"]] = None
    _clst_pts: Optional[Float[torch.Tensor, "n_scene n_cand 3"]] = field(default=None, repr=False)
    _approach_dir: Optional[Float[torch.Tensor, "n_scene n_cand 3"]] = field(default=None, repr=False)

    @property
    def clst_pts(self) -> Float[torch.Tensor, "n_scene n_cand 3"]:
        if self._clst_pts is None:
            _, _, self._clst_pts = self.scene_geom.query_target_object_signed_distances(self.tl)
        return self._clst_pts

    @property
    def approach_dir(self) -> Float[torch.Tensor, "n_scene n_cand 3"]:
        """Unit vector from the nearest object point out to the hand."""
        if self._approach_dir is None:
            self._approach_dir = F.normalize(self.tl - self.clst_pts, p=2, dim=-1)
        return self._approach_dir

    def require_qpos(self, filter_name: str) -> Float[torch.Tensor, "n_scene n_cand dof"]:
        assert self.qpos is not None, f"{filter_name} needs qpos; this stage has candidates only"
        return self.qpos


class Filter(ABC):
    """Thresholds in __init__, `__call__(cands) -> keep mask`. Same shape for every stage."""

    name: str

    @abstractmethod
    def __call__(self, cands: Candidates) -> Bool[torch.Tensor, "n_scene n_cand"]:
        raise NotImplementedError


class Sdf(Filter):
    """Hand position is outside the scene geometry (not buried in the table or object)."""

    name = "sdf"

    def __call__(self, cands: Candidates) -> Bool[torch.Tensor, "n_scene n_cand"]:
        sdf, _, _ = cands.scene_geom.query_scene_signed_distances(cands.tl)
        return sdf > 0.0


class RestPose(Filter):
    """Arm qpos stays within `max_dist` (L2 over the arm joints) of the robot's rest pose."""

    name = "rest_pose"

    def __init__(self, max_dist: float):
        self.max_dist = max_dist

    def __call__(self, cands: Candidates) -> Bool[torch.Tensor, "n_scene n_cand"]:
        qpos = cands.require_qpos(self.name)
        arm_indices = cands.robot.arm_joint_indices
        init_arm_qpos = cands.robot.init_qpos.to(qpos.device)[..., arm_indices]
        return torch.norm(qpos[..., arm_indices] - init_arm_qpos, dim=-1) < self.max_dist


class CollisionFree(Filter):
    """No collision sphere of the posed robot penetrates the scene."""

    name = "collision_free"

    def __call__(self, cands: Candidates) -> Bool[torch.Tensor, "n_scene n_cand"]:
        qpos = cands.require_qpos(self.name)
        n_scene, n_cand = qpos.shape[:2]
        link_poses = cands.robot.forward_kinematics_torch(qpos, root_poses=cands.robot.root_pose_mat)
        spheres = cands.robot.get_collision_spheres(link_poses)
        spheres_flat = spheres.reshape(n_scene, -1, spheres.shape[-1])
        sdf, _, _ = cands.scene_geom.query_scene_signed_distances(spheres_flat[..., :3])
        penetration = (spheres_flat[..., 3] - sdf).reshape(n_scene, n_cand, -1)
        return penetration.max(dim=-1).values < 0.0


class Ori(Filter):
    """`hand_coord_spec` axis within `max_angle_deg` of `target` in the robot base frame (world for floating hands)."""

    name = "ori"

    def __init__(self, hand_axis: HandAxis, target: Tuple[float, float, float], max_angle_deg: float):
        self.hand_axis = hand_axis
        self.target = target
        self.max_angle_deg = max_angle_deg

    def __call__(self, cands: Candidates) -> Bool[torch.Tensor, "n_scene n_cand"]:
        robot = cands.robot
        device = cands.rot.device
        rot = cands.rot
        if robot.config.base_type == "fixed":
            eye = torch.eye(4, device=device)
            link_poses = robot.forward_kinematics_torch(cands.require_qpos(self.name), root_poses=eye)
            rot = link_poses[..., robot.link_names.index(robot.config.hand_base_link_name), :3, :3]
        axis = AXIS_TO_VEC[getattr(hand_coord_spec(robot.config), f"{self.hand_axis}_axis")]
        local = torch.tensor(axis, device=device, dtype=torch.float32)
        target = F.normalize(torch.tensor(self.target, device=device, dtype=torch.float32), dim=-1)
        cos_angle = ((rot @ local) * target).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cos_angle)) <= self.max_angle_deg


class Pen(Filter):
    """Hand collision spheres overlap the target object by at most `threshold` metres."""

    name = "pen"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, cands: Candidates) -> Bool[torch.Tensor, "n_scene n_cand"]:
        qpos = cands.require_qpos(self.name)
        depth = compute_penetration_depth(cands.scene_geom.scene_descriptions, qpos, cands.robot)
        return depth <= self.threshold


class ContactAvoid(Filter):
    """Contact links sit closer to the target than their paired avoid links."""

    name = "contact_avoid"

    @staticmethod
    def load_link_points(path: str, device: Device) -> Dict[str, Float[torch.Tensor, "n_points 3"]]:
        """One representative point per link: the recorded point nearest the link's point centroid."""
        with open(path, "r", encoding="utf-8") as file:
            raw_points = json.load(file)

        link_points = {}
        for link_name, points in raw_points.items():
            if len(points) > 0:
                points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
                center = points_tensor.mean(dim=0, keepdim=True)
                point_index = torch.argmin(torch.linalg.vector_norm(points_tensor - center, dim=-1))
                link_points[link_name] = points_tensor[point_index : point_index + 1]
        return link_points

    def __call__(self, cands: Candidates) -> Bool[torch.Tensor, "n_scene n_cand"]:
        grasp_qpos = cands.require_qpos(self.name)
        robot = cands.robot
        assert robot.config.contact_points_path is not None or robot.config.contact_candidates_config is not None, (
            "contact avoid filtering requires contact_points_path or contact_candidates_config"
        )
        assert robot.config.contact_avoid_points_path is not None, (
            "contact avoid filtering requires contact avoid points"
        )
        assert grasp_qpos.ndim == 3, "contact avoid filtering expects padded grasp qpos"

        device = grasp_qpos.device
        if robot.config.contact_points_path is not None:
            contact_points_by_link = self.load_link_points(robot.config.contact_points_path, device)
        else:
            contact_candidates, contact_link_indices = GraspRobot.get_contact_candidates(robot.config, device=device)
            contact_points_by_link = {}
            for link_index in torch.unique(contact_link_indices).tolist():
                link_points = contact_candidates[contact_link_indices == link_index]
                center = link_points.mean(dim=0, keepdim=True)
                point_index = torch.argmin(torch.linalg.vector_norm(link_points - center, dim=-1))
                contact_points_by_link[robot.link_names[link_index]] = link_points[point_index : point_index + 1]
        avoid_points_by_link = self.load_link_points(robot.config.contact_avoid_points_path, device)
        link_names = sorted(set(contact_points_by_link) & set(avoid_points_by_link))
        if len(link_names) == 0:
            raise ValueError("No links have both contact points and contact avoid points")

        link_indices = {link_name: robot.link_names.index(link_name) for link_name in link_names}
        scene_geom = cands.scene_geom
        keep_mask = torch.ones(grasp_qpos.shape[:2], dtype=torch.bool, device=device)
        num_scenes, num_grasps = grasp_qpos.shape[:2]
        filter_batch_size = 2048

        for start_index in range(0, num_grasps, filter_batch_size):
            chunk_grasp_qpos = grasp_qpos[:, start_index : start_index + filter_batch_size]
            chunk_size = chunk_grasp_qpos.shape[1]
            link_poses = robot.forward_kinematics_torch(chunk_grasp_qpos.reshape(num_scenes * chunk_size, -1))
            link_poses = link_poses.reshape(num_scenes, chunk_size, *link_poses.shape[1:])
            chunk_keep_mask = torch.ones((num_scenes, chunk_size), dtype=torch.bool, device=device)

            for link_name in link_names:
                link_pose = link_poses[:, :, link_indices[link_name]]
                rotation = link_pose[:, :, :3, :3]
                translation = link_pose[:, :, :3, 3]
                contact_points = contact_points_by_link[link_name]
                avoid_points = avoid_points_by_link[link_name]
                contact_world = contact_points.view(1, 1, -1, 3) @ rotation.transpose(-1, -2)
                avoid_world = avoid_points.view(1, 1, -1, 3) @ rotation.transpose(-1, -2)
                contact_world = contact_world + translation[:, :, None]
                avoid_world = avoid_world + translation[:, :, None]
                query_points = torch.cat([contact_world, avoid_world], dim=2)
                signed_distances, _, _ = scene_geom.query_target_object_signed_distances(
                    query_points.reshape(num_scenes, chunk_size * query_points.shape[2], 3)
                )
                signed_distances = signed_distances.reshape(num_scenes, chunk_size, query_points.shape[2])
                contact_distances = signed_distances[:, :, : contact_world.shape[2]].abs().min(dim=2).values
                avoid_distances = signed_distances[:, :, contact_world.shape[2] :].abs().min(dim=2).values
                chunk_keep_mask &= contact_distances <= avoid_distances

            keep_mask[:, start_index : start_index + chunk_size] = chunk_keep_mask

        return keep_mask


def compute_penetration_depth(
    scene_descriptions: List[SceneDescription],
    grasp_qpos: Float[torch.Tensor, "n_scene num_grasps dof"],
    robot_helper: GraspRobot,
) -> Float[torch.Tensor, "n_scene num_grasps"]:
    """Max overlap depth (meters) of the hand's collision spheres into the target object, per grasp."""
    device = grasp_qpos.device
    n_scene, num_grasps = grasp_qpos.shape[:2]
    scene_geom = SceneGeom(scene_descriptions, device=device)
    link_poses = robot_helper.forward_kinematics_torch(grasp_qpos)
    spheres = robot_helper.get_collision_spheres(link_poses=link_poses)
    n_sphere = spheres.shape[-2]
    sdf, _, _ = scene_geom.query_target_object_signed_distances(
        spheres[..., :3].reshape(n_scene, num_grasps * n_sphere, 3)
    )
    pen = (spheres[..., 3] - sdf.reshape(n_scene, num_grasps, n_sphere)).clamp(min=0.0)
    return pen.max(dim=-1).values


def compute_self_collision_depth(
    grasp_qpos: Float[torch.Tensor, "n_scene num_grasps dof"],
    robot_helper: GraspRobot,
) -> Float[torch.Tensor, "n_scene num_grasps"]:
    """Max overlap depth (meters) of the hand's own collision spheres per grasp, on SelfCollisionTask's active pairs."""
    geom_link = robot_helper.canonical_collision_spheres[1].cpu().numpy()
    pairs = compute_active_collision_pairs(robot_helper.spec, geom_link_indices=geom_link)
    ignored = {(int(a), int(b)) for a, b in robot_helper.self_collision_ignored_pairs("sphere")}
    if ignored and len(pairs) > 0:
        ignored_both = ignored | {(j, i) for i, j in ignored}
        keep = [(int(geom_link[i]), int(geom_link[j])) not in ignored_both for i, j in pairs]
        pairs = pairs[np.asarray(keep, dtype=bool)]
    if len(pairs) == 0:
        # A two-jaw gripper's links are all mutually adjacent, so nothing is left to check and depth is zero.
        return torch.zeros(grasp_qpos.shape[:-1], device=grasp_qpos.device)
    i = torch.as_tensor(pairs[:, 0], device=grasp_qpos.device)
    j = torch.as_tensor(pairs[:, 1], device=grasp_qpos.device)
    spheres = robot_helper.get_collision_spheres(link_poses=robot_helper.forward_kinematics_torch(grasp_qpos))
    dist = torch.linalg.vector_norm(spheres[..., i, :3] - spheres[..., j, :3], dim=-1)
    overlap = (spheres[..., i, 3] + spheres[..., j, 3] - dist).clamp(min=0.0)
    return overlap.max(dim=-1).values
