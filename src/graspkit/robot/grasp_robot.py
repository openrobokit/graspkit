# pyright: reportArgumentType=false
"""Grasp-pipeline torch facade (FK/IK, joint partitions, collision spheres, contact-point candidates, root pose)."""

import json
import logging
from collections import defaultdict
from functools import cached_property, lru_cache
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import warp as wp
import yaml
from jaxtyping import Bool, Float, Int
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.lie.se3 import se3_to_matrix
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.robo.robot import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.utils.ee_coord_utils import AXIS_TO_VEC, GripperCoordinateSpec, HandCoordinateSpec
from robokit.xform.utils import rotation_6d_to_matrix
from robokit.xform.warp.torch_wrappers import (
    euler_angles_to_matrix,
    pose7_to_tf_mat,
    rot_tl_to_tf_mat,
    transform_points,
)
from torch.types import Device

from graspkit.configs.robot_config import RobotConfig

logger = logging.getLogger("graspkit")


class GraspRobot(Robot):
    """Robot equipped with the grasp-pipeline torch facade and contact-point candidates."""

    def __init__(self, spec: RobotSpec, config: Optional[RobotConfig] = None, device: Device = "cpu"):
        super().__init__(spec)
        self.config = config
        self.device = device

    @property
    def actuated_joint_names(self) -> "list[str]":
        return self.spec.actuated_joint_names

    @staticmethod
    @lru_cache(maxsize=None)
    def from_config(
        config: RobotConfig,
        base_link_name: Optional[str] = None,
        ee_link_names: Optional[Union[Tuple[str, ...], str]] = None,
        load_meshes: bool = False,
        load_collision_spheres: bool = False,
        device: Device = "cpu",
        prune: bool = True,
        self_collision_representation: Literal["sphere", "capsule"] = "sphere",
    ) -> "GraspRobot":
        """Build (and cache) a GraspRobot from RobotConfig; pruned to ee_link_names, prune=False for the full robot."""
        wp.init()
        if prune and ee_link_names is None:
            ee_link_names = config.ee_link_names
        spec = RobotSpec.parse(
            robot_description_or_path=config.urdf_or_mjcf_path,
            load_meshes=load_meshes,
            mesh_dir=config.mesh_dir,
            load_collision_spheres=load_collision_spheres,
            collision_spheres_path=config.collision_spheres_path if load_collision_spheres else None,
            base_link_name=base_link_name,
            ee_link_names=ee_link_names,
            keep_mjcf_world_link=config.keep_world_link,
        )
        robot = GraspRobot(spec, config=config, device=device)
        spec.self_collision_ignored_pairs = robot.self_collision_ignored_pairs(self_collision_representation)
        return robot

    @lru_cache(maxsize=None)
    def self_collision_ignored_pairs(self, representation: Literal["sphere", "capsule"]) -> Int[np.ndarray, "pairs 2"]:
        """Load representation-specific pairs against this robot's pruned link names."""
        path = self.config.get_self_collision_ignore_path(representation)
        if path is None:
            return np.empty((0, 2), dtype=np.int32)
        return RobotSpec._load_self_collision_ignore(self.spec.link_names, path)

    # ------------------- forward / inverse kinematics -------------------
    @cached_property
    def root_pose_7d(self) -> Float[torch.Tensor, "7"]:
        return torch.tensor((*self.config.init_position, *self.config.init_quaternion), device=self.device)

    @cached_property
    def root_pose_mat(self) -> Float[torch.Tensor, "4 4"]:
        return pose7_to_tf_mat(self.root_pose_7d)

    @cached_property
    def actuated_joint_limits_torch(self) -> Float[torch.Tensor, "num_dofs 2"]:
        return torch.tensor(self.spec.actuated_joint_limits, device=self.device, dtype=torch.float32)

    @cached_property
    def init_qpos(self) -> Float[torch.Tensor, "num_dofs"]:  # noqa: F821
        # config init_qpos is full-robot order; on a pruned chain gather it by joint name to size it correctly
        values = self.config.init_qpos
        if len(values) != self.num_actuated_joints:
            full_names = GraspRobot.from_config(self.config, prune=False).actuated_joint_names
            values = tuple(values[full_names.index(n)] for n in self.actuated_joint_names)
        return torch.tensor(values, device=self.device)

    def forward_kinematics_torch(
        self,
        variables: Float[torch.Tensor, "... num_dofs"],
        root_poses: Optional[Float[torch.Tensor, "... 4 4"]] = None,
    ) -> Float[torch.Tensor, "... num_links 4 4"]:
        """Torch FK. Two cases:
        1. floating base: variables packs free qpos (xyz + rxyz or xyz + 6d) then actuated qpos
           (root_poses must not be provided).
        2. fixed base: variables is just qpos (root_poses optional).
        """

        if self.config.base_type == "floating":
            if root_poses is not None:
                raise ValueError("root_poses should not be provided if the robot has a floating base")
            if variables.shape[-1] == self.num_actuated_joints + 6:
                root_poses = rot_tl_to_tf_mat(
                    rot_mat=euler_angles_to_matrix(variables[..., 3:6], "rxyz"), tl=variables[..., :3]
                )
                qpos = variables[..., 6:]
            elif variables.shape[-1] == self.num_actuated_joints + 9:
                root_poses = rot_tl_to_tf_mat(rot_mat=rotation_6d_to_matrix(variables[..., 3:9]), tl=variables[..., :3])
                qpos = variables[..., 9:]
            else:
                raise ValueError(f"Invalid shape of variables: {variables.shape}")
        else:  # fixed base
            qpos = variables
            if root_poses is None:  # use root_poses provided in the config
                root_poses = self.root_pose_mat

        joint_limits = self.actuated_joint_limits_torch.to(qpos.device)
        qpos = torch.clamp(qpos, min=joint_limits[:, 0], max=joint_limits[:, 1])

        root_poses = root_poses.to(device=qpos.device, dtype=torch.float32)
        return self.forward_kinematics_via_matrix_torch(qpos, T_world_base=root_poses).clone()

    @staticmethod
    @lru_cache(maxsize=8)
    def _get_ik_helper(
        config: RobotConfig,
        batch_size: int,
        device_str: str,
        use_score_rest: bool = False,
        score_position_threshold: float = 0.005,
        score_orientation_threshold: float = 0.05,
        locked_joint_indices: Tuple[int, ...] = (),
    ) -> IKSolver:
        wp.init()
        arm_robot = GraspRobot.from_config(
            config, base_link_name=config.arm_base_link_name, ee_link_names=config.hand_base_link_name
        )
        ee_link_name = config.hand_base_link_name or arm_robot.link_names[-1]
        solver_config = PopulationSolverConfig(  # 128 seeds for 16 LM iterations, the best 16 for 24 more, then one
            num_iters=16 + 24,
            select_at=[16, 16 + 24],
            num_candidates=[128, 16, 1],
            optimizer=[LMOptimizerConfig(lm_lambda=10.0), LMOptimizerConfig(lm_lambda=1.0), LMOptimizerConfig()],
            cuda_graph_mode="iter",  # "full" caches per-call addresses; replayed stale -> CUDA illegal memory access
        )
        target = {"frame_index": IKContext.TARGET_LINKS, "T_world_target": IKContext.TARGET}
        ik_config = (
            IKSolverConfig(solver=solver_config)
            .add(PositionTask(**target, weight=2.0))
            .add(RotationTask(**target, weight=0.2))
            .add(PositionLimit(weight=50.0))
        )
        if use_score_rest:
            ik_config.add_score(PositionTask(**target, weight=2.0))
            ik_config.add_score(RotationTask(**target, weight=0.2))
            ik_config.add_score(ConfigurationTask(rest_q=IKContext.REST_Q, weight=0.1, base_weight=0.0))
        ik_helper = IKSolver(ik_config, arm_robot, ee_link_name, device_str)
        ik_helper.warmup(batch_size)
        # lock baked into the cache key (mask built once per helper); locked indices are chain DOF, not arm_robot's own
        if locked_joint_indices:
            locked = set(locked_joint_indices)
            chain_names = GraspRobot.from_config(config).actuated_joint_names
            mask = [0.0 if chain_names.index(jn) in locked else 1.0 for jn in arm_robot.actuated_joint_names]
            ik_helper.set_active_joint_mask(mask)
        return ik_helper

    def inverse_kinematics(
        self,
        target_poses: Float[torch.Tensor, "... 4 4"],
        init_qpos: Optional[Float[torch.Tensor, "... num_arm_dofs"]] = None,
        position_tolerance: float = 0.05,
        orientation_tolerance: float = 0.3,
        init_sample_range: float = 0.5,
        use_score_rest: bool = False,
        locked_joint_indices: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[Float[torch.Tensor, "... num_arm_dofs"], Bool[torch.Tensor, "..."]]:
        with torch.no_grad():
            original_shape = target_poses.shape[:-2]
            total_batch = int(target_poses.numel() // 16)
            flat_poses = target_poses.reshape(total_batch, 4, 4)

            init_q = None
            if init_qpos is not None:
                init_q = init_qpos.reshape(-1, init_qpos.shape[-1])
                if init_q.shape[0] == 1:
                    init_q = init_q.expand(total_batch, -1).contiguous()
                elif init_q.shape[0] != total_batch and total_batch % init_q.shape[0] == 0:
                    repeat_factor = total_batch // init_q.shape[0]
                    init_q = init_q.repeat_interleave(repeat_factor, dim=0)

            device_str = f"cuda:{target_poses.device.index}" if target_poses.is_cuda else "cpu"
            ik_helper = GraspRobot._get_ik_helper(
                self.config,
                total_batch,
                device_str,
                use_score_rest,
                score_position_threshold=position_tolerance,
                score_orientation_threshold=orientation_tolerance,
                locked_joint_indices=locked_joint_indices or (),
            )

            result = ik_helper.solve_torch(
                flat_poses,
                init_q=init_q,
                rest_q=init_q,
                init_sample_range=init_sample_range,
            )

            num_arm_dofs = ik_helper.robot.spec.num_actuated_joints
            result_q = result.q.reshape(*original_shape, num_arm_dofs).to(target_poses.device)

            # Compute errors via FK on the result
            ee_idx = ik_helper.target_link_indices[0]
            fk_poses = ik_helper.robot.forward_kinematics_via_matrix_torch(result.q)
            ee_poses = fk_poses[..., ee_idx, :, :]
            position_error = torch.norm(ee_poses[..., :3, 3] - flat_poses[..., :3, 3], dim=-1)
            R_result = ee_poses[..., :3, :3]
            R_target = flat_poses[..., :3, :3]
            trace = (R_result * R_target).sum(dim=(-2, -1))
            orientation_error = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
            success_mask = (position_error < position_tolerance) & (orientation_error < orientation_tolerance)
            success_mask = success_mask.reshape(*original_shape).to(target_poses.device)

            return result_q, success_mask

    # ------------------- joint partitions -------------------
    @cached_property
    def _arm_hand_joint_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actuated-joint indices of arm and hand subtree (robot order); without hand_base, both are all joints."""
        if self.config.hand_base_link_name is None:
            all_indices = torch.arange(self.num_actuated_joints, device=self.device)
            return all_indices, all_indices
        link_parents = self.spec.link_parent_joint_indices
        joint_parents = self.spec.parent_joint_indices

        def ancestors(joint_idx: int) -> set[int]:
            chain = set()
            while joint_idx != -1:
                chain.add(joint_idx)
                joint_idx = int(joint_parents[joint_idx])
            return chain

        hand_parent = int(link_parents[self.link_names.index(self.config.hand_base_link_name)])
        arm = ancestors(hand_parent)
        if self.config.arm_base_link_name is not None:
            arm -= ancestors(int(link_parents[self.link_names.index(self.config.arm_base_link_name)]))
        arm_indices, hand_indices = [], []
        for j in range(len(self.spec.joint_names)):
            actuated_idx = int(self.spec.actuated_joint_indices[j])
            if actuated_idx < 0:
                continue
            if j in arm:
                arm_indices.append(actuated_idx)
            elif hand_parent == -1 or hand_parent in ancestors(j):
                hand_indices.append(actuated_idx)
        return (
            torch.tensor(arm_indices, dtype=torch.long, device=self.device),
            torch.tensor(hand_indices, dtype=torch.long, device=self.device),
        )

    @property
    def arm_joint_indices(self) -> torch.Tensor:
        return self._arm_hand_joint_indices[0]

    @property
    def hand_joint_indices(self) -> torch.Tensor:
        return self._arm_hand_joint_indices[1]

    @cached_property
    def ignore_joint_indices(self) -> torch.Tensor:
        """Actuated joints that are neither arm nor hand (complement of arm ∪ hand indices)."""
        keep = set(self.arm_joint_indices.tolist()) | set(self.hand_joint_indices.tolist())
        return torch.tensor(
            [i for i in range(self.num_actuated_joints) if i not in keep], dtype=torch.long, device=self.device
        )

    # ------------------- collision spheres -------------------
    @cached_property
    def canonical_collision_spheres(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Link-local collision spheres from collision_spheres_path: (num_spheres,4) centers+radius, link indices."""
        assert self.config is not None and self.config.collision_spheres_path is not None
        with open(self.config.collision_spheres_path, "r") as f:
            collision_spheres_data = yaml.safe_load(f)["collision_spheres"]
        sphs, link_indices = [], []
        for link_name in set(self.link_names) & set(collision_spheres_data.keys()):
            spheres = collision_spheres_data[link_name]
            sphs.extend([s["center"] + [s["radius"]] for s in spheres])
            link_indices.extend([self.link_names.index(link_name)] * len(spheres))
        return (
            torch.tensor(sphs, dtype=torch.float, device=self.device),
            torch.tensor(link_indices, dtype=torch.long, device=self.device),
        )

    def get_collision_spheres(
        self, link_poses: Float[torch.Tensor, "... num_links 4 4"]
    ) -> Float[torch.Tensor, "... num_spheres 4"]:
        spheres, link_indices = self.canonical_collision_spheres
        tf_mat = torch.index_select(link_poses, dim=-3, index=link_indices)
        sphere_centers = transform_points(spheres[..., :3].unsqueeze(-2), tf_mat).squeeze(-2)
        sphere_radius = spheres[..., 3].unsqueeze(-1).expand_as(sphere_centers[..., :1])
        return torch.cat([sphere_centers, sphere_radius], dim=-1)

    # ------------------- contact candidates -------------------
    def get_contact_points(
        self,
        link_poses: Float[torch.Tensor, "... num_links 4 4"],
        contact_point_indices: Int[torch.Tensor, "... num_contact_points"],
    ) -> Float[torch.Tensor, "... num_contact_points 3"]:
        candidates, candidate_link_indices = GraspRobot.get_contact_candidates(self.config, self.device)
        canonical_contact_points = candidates[contact_point_indices]
        link_indices = candidate_link_indices[contact_point_indices]
        tf_mat = torch.gather(link_poses, dim=-3, index=link_indices[..., None, None].expand(*link_indices.shape, 4, 4))
        return transform_points(canonical_contact_points.unsqueeze(-2), tf_mat).squeeze(-2)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_contact_candidates(config: RobotConfig, device: Device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Canonical contact candidates: (N,3) link-local centers, (N,) link indices in the chain robot's link order."""
        full_robot = GraspRobot.from_config(config)
        if config.contact_points_path is not None:
            with open(config.contact_points_path, "r") as f:
                contact_points = json.load(f)
        else:
            hand_robot = Robot.load(
                robot_description_or_path=config.urdf_or_mjcf_path,
                mesh_dir=config.mesh_dir,
                base_link_name=config.hand_base_link_name,
                load_meshes=True,
            )
            num_per_link = config.contact_candidates_config
            if num_per_link is None:
                logger.warning("No contact candidates configuration provided, sample 10 points for each hand link")
            sample_q = (
                np.asarray(config.contact_sampling_qpos, dtype=np.float32)
                if config.contact_sampling_qpos is not None
                else np.zeros(hand_robot.num_actuated_joints, dtype=np.float32)
            )
            state = hand_robot.forward_kinematics(hand_robot.state(q=sample_q))
            link_poses = se3_to_matrix(state.T_world_link).numpy().reshape(-1, 4, 4)
            thumb_links = set(config.thumb_link_names or ())
            spec = hand_coord_spec(config)
            contact_points = defaultdict(list)
            for link_name in hand_robot.spec.link_names:
                if num_per_link is not None and link_name not in num_per_link:
                    continue
                n = 10 if num_per_link is None else num_per_link[link_name]
                link_mesh = hand_robot.spec.get_link_mesh(link_name, mode="collision", return_mesh=True)
                if len(link_mesh.faces) == 0:
                    continue
                link_idx = hand_robot.spec.link_names.index(link_name)
                dense_pts, face_ids = link_mesh.sample(100 * n, return_index=True)
                rot, tl = link_poses[link_idx, :3, :3], link_poses[link_idx, :3, 3]
                normals_world = link_mesh.face_normals[face_ids] @ rot.T
                if isinstance(config.ee_coord_spec, GripperCoordinateSpec):  # jaw pads face the closing (outward) axis
                    out = spec.palm_forward_vec
                    radial = dense_pts @ rot.T + tl
                    radial = radial - np.outer(radial @ out, out)
                    facing = -radial / np.linalg.norm(radial, axis=-1, keepdims=True)
                    pts = dense_pts[(normals_world * facing).sum(-1) > 0.7]
                else:
                    facing_dir = spec.four_fingers_up_vec if link_name in thumb_links else spec.palm_forward_vec
                    pts = dense_pts[(normals_world @ facing_dir) > 0.7]
                if len(pts) == 0:
                    continue
                if len(pts) > n:
                    # farthest-point sampling
                    keep_idx = np.zeros(n, dtype=np.int64)
                    distances = np.full(len(pts), np.inf, dtype=np.float32)
                    for i in range(1, n):
                        d = np.sum((pts - pts[keep_idx[i - 1]]) ** 2, axis=-1)
                        distances = np.minimum(distances, d)
                        keep_idx[i] = int(np.argmax(distances))
                    pts = pts[keep_idx]
                contact_points[link_name].extend(pts)
        if config.contact_link_names is not None:
            contact_points = {k: v for k, v in contact_points.items() if k in config.contact_link_names}
        contact_candidates = []
        contact_candidates_link_indices = []
        for link_name, points in contact_points.items():
            contact_candidates.extend(points)
            contact_candidates_link_indices.extend([full_robot.link_names.index(link_name)] * len(points))
        contact_candidates = torch.from_numpy(np.asarray(contact_candidates, dtype=np.float32)).to(device)
        contact_candidates_link_indices = torch.tensor(contact_candidates_link_indices, dtype=torch.long, device=device)
        if contact_candidates.shape[0] == 0:
            raise ValueError("No contact candidates found")
        return contact_candidates, contact_candidates_link_indices


@lru_cache(maxsize=None)
def hand_coord_spec(config: RobotConfig) -> HandCoordinateSpec:
    """`config.ee_coord_spec` in the hand frame (`hand_base_link_name`, else the root), meaning what grasps need:
    palm_forward faces the object. A gripper's jaws (outward) are its palm; its four_fingers_up is outward x opening.
    """
    spec = config.ee_coord_spec
    assert spec is not None, f"robot config {config.name!r} has no ee_coord_spec"
    if isinstance(spec, GripperCoordinateSpec):
        out, opening = np.asarray(AXIS_TO_VEC[spec.outward_axis]), np.asarray(AXIS_TO_VEC[spec.opening_axis])
        axes = (out, np.cross(out, opening))
    else:
        axes = (spec.palm_forward_vec, spec.four_fingers_up_vec)
    if config.ee_coord_link_name is not None:  # rotate from robokit's mount link into the hand frame
        robot = GraspRobot.from_config(config, prune=False)
        floating = 6 if config.base_type == "floating" else 0
        poses = robot.forward_kinematics_torch(torch.zeros(floating + robot.num_actuated_joints))
        hand = robot.link_names.index(config.hand_base_link_name or robot.link_names[0])
        rot = (torch.linalg.inv(poses[hand]) @ poses[robot.link_names.index(config.ee_coord_link_name)])[:3, :3]
        axes = tuple(rot.numpy() @ np.asarray(v, dtype=np.float32) for v in axes)
    names = {tuple(v): k for k, v in AXIS_TO_VEC.items()}
    rounded = [np.round(v).astype(int) for v in axes]
    assert all(np.abs(v - r).max() < 0.05 for v, r in zip(axes, rounded)), (
        f"{config.name}: hand frame is not axis-aligned"
    )
    return HandCoordinateSpec(*(names[tuple(r.tolist())] for r in rounded))


__all__ = ["GraspRobot", "hand_coord_spec"]
