"""Grasp-init samplers: shared pipeline (points -> directions -> rotations -> IK -> collision -> hand jitter)."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float
from robokit.utils.ee_coord_utils import ee_coord_conversion
from robokit.xform.utils import look_at_rotation
from robokit.xform.warp.torch_wrappers import (
    euler_angles_to_matrix,
    inverse_tf_mat,
    matrix_to_euler_angles,
    rot_tl_to_tf_mat,
)
from torch.types import Device

from graspkit.configs.experiment_config import GraspOptimConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.grasp_robot import GraspRobot, hand_coord_spec
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.grasp_filters import (
    Candidates,
    CollisionFree,
    Filter,
    Ori,
    RestPose,
    Sdf,
)
from graspkit.utils.tensor_utils import atleast_nd
from graspkit.utils.xform_utils import OPENGL

logger = logging.getLogger("graspkit")


def _batch_sample(
    x: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    mask: torch.Tensor,
    num_samples: int,
    device: torch.device,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    selected_indices = []
    for i in range(len(mask)):
        valid_indices = torch.nonzero(mask[i]).squeeze(1)
        if valid_indices.shape[0] == 0:
            logger.warning(f"scene {i}: no valid candidates, falling back to random sampling from all")
            selected_indices.append(torch.randint(0, mask.shape[1], (num_samples,), dtype=torch.long, device=device))
        elif valid_indices.shape[0] < num_samples:
            logger.warning(
                f"scene {i}: only {valid_indices.shape[0]}/{num_samples} valid candidates, sampling with replacement"
            )
            selected_indices.append(
                valid_indices[torch.randint(0, valid_indices.shape[0], (num_samples,), device=device)]
            )
        else:
            perm = torch.randperm(valid_indices.shape[0])
            selected_indices.append(valid_indices[perm[:num_samples]])
    selected_indices = torch.stack(selected_indices)
    if isinstance(x, torch.Tensor):
        return torch.gather(
            x,
            1,
            atleast_nd(selected_indices, x.ndim, add_dim_to_front=False).expand(-1, -1, *x.shape[2:]),
        )
    return tuple(
        torch.gather(
            t,
            1,
            atleast_nd(selected_indices, t.ndim, add_dim_to_front=False).expand(-1, -1, *t.shape[2:]),
        )
        for t in x
    )


class Sampler(ABC):
    """Base sampler: shared building blocks plus the abstract `sample` entry point."""

    def __init__(self, cfg: GraspOptimConfig, robot_config: RobotConfig, robot_helper: GraspRobot, device: Device):
        self.cfg = cfg
        self.robot_config = robot_config
        self.robot_helper = robot_helper
        self.device = device
        # Raw per-candidate masks + indexed translations/qpos from the last sample() call; consumers reduce them.
        self.diagnostics = {}
        # Pre-IK filters see poses only; post-IK ones need qpos. Masks kept per-filter for viz.
        self.pre_ik_filters: list[Filter] = [Sdf()]
        self.post_ik_filters: list[Filter] = [RestPose(cfg.init_rest_pose_max_dist), CollisionFree()]
        if cfg.hand_axis is not None:
            ori = Ori(cfg.hand_axis, cfg.hand_axis_target, cfg.filter_hand_axis_max_angle_deg)
            if robot_config.base_type == "floating":  # no IK: the candidate's rotation is the grasp's
                self.pre_ik_filters.append(ori)
            else:
                self.post_ik_filters.append(ori)

    @abstractmethod
    def sample(
        self, scene_geom: SceneGeom, num_samples_per_scene: int
    ) -> Float[torch.Tensor, "n_scene n_samples num_dofs"]:
        raise NotImplementedError("Subclasses must implement this method")

    # ── shared building blocks ────────────────────────────────────────────────

    def _sample_lookat(
        self,
        scene_geom: SceneGeom,
        num_samples_per_scene: int,
        num_points_per_sample: int,
        *,
        center_on_target: bool,
        fold_upper: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample points -> translations + look-at rotations; returns (clst_pts, approach_dir, tl, lookat_rot)."""
        device = self.device
        cfg = self.cfg

        # 1. sample points (WBC centers the [-1,1] cube on target so approach directions cover its full sphere)
        rand_points = torch.rand((len(scene_geom), num_points_per_sample * num_samples_per_scene, 3), device=device)
        rand_points = rand_points * 2.0 - 1.0
        if center_on_target:
            target_positions = torch.tensor(
                [desc.meshes[desc.target_actor_id].position for desc in scene_geom.scene_descriptions], device=device
            )
            rand_points = rand_points + target_positions[:, None, :]

        # 2. find the closest point to the object
        _, _, clst_pts = scene_geom.query_target_object_signed_distances(rand_points)

        # 3. sample distance + translation (WBC folds to upper hemisphere; hand never comes from under the table)
        distance = cfg.distance_lower + (cfg.distance_upper - cfg.distance_lower) * torch.rand_like(
            rand_points[..., 0:1]
        )
        approach_dir = F.normalize(rand_points - clst_pts, p=2, dim=-1)
        if fold_upper:
            approach_dir[..., 2] = approach_dir[..., 2].abs()
        tl = clst_pts + distance * approach_dir

        # 4. sample rotation jitter + look-at hand rotation
        lookat_rot_opengl = look_at_rotation(tl, clst_pts, torch.tensor([0.0, 0.0, 1.0], device=device))
        z_theta = cfg.front_theta_lower + (cfg.front_theta_upper - cfg.front_theta_lower) * torch.rand_like(tl[..., 0])
        y_theta = cfg.side_theta_lower + (cfg.side_theta_upper - cfg.side_theta_lower) * torch.rand_like(tl[..., 0])
        x_theta = cfg.side_theta_lower + (cfg.side_theta_upper - cfg.side_theta_lower) * torch.rand_like(tl[..., 0])
        jitter_rot_opengl = euler_angles_to_matrix(torch.stack([z_theta, y_theta, x_theta], dim=-1), axes="szyx")
        lookat_rot = (
            lookat_rot_opengl
            @ jitter_rot_opengl
            @ ee_coord_conversion(hand_coord_spec(self.robot_config), OPENGL, return_tensors="pt").to(device)
        )
        return clst_pts, approach_dir, tl, lookat_rot

    def _run_pre_ik_filters(
        self,
        scene_geom: SceneGeom,
        tl: torch.Tensor,
        lookat_rot: torch.Tensor,
        clst_pts: torch.Tensor,
        approach_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Run pose-only filters into the first-resample mask; clst_pts/approach_dir passed in, not re-derived."""
        cands = Candidates(
            tl=tl,
            rot=lookat_rot,
            robot=self.robot_helper,
            scene_geom=scene_geom,
            _clst_pts=clst_pts,
            _approach_dir=approach_dir,
        )
        self.diagnostics = {f.name: f(cands) for f in self.pre_ik_filters}
        first_mask = torch.ones_like(next(iter(self.diagnostics.values())))
        for mask in self.diagnostics.values():
            first_mask = first_mask & mask
        self.diagnostics["geom_combined"] = first_mask
        self.diagnostics = {k: v.cpu() for k, v in self.diagnostics.items()}
        self.diagnostics["geom_tl"] = tl.cpu()
        return first_mask

    def _solve_ik_and_hand(
        self,
        scene_geom: SceneGeom,
        lookat_rot: torch.Tensor,
        tl: torch.Tensor,
        first_mask: torch.Tensor,
        num_samples_per_scene: int,
    ) -> Float[torch.Tensor, "n_scene n_samples num_dofs"]:
        """Batched arm IK + collision filtering on the masked poses, then hand-joint jitter."""
        robot_helper = self.robot_helper
        device = self.device
        cfg = self.cfg

        # 7. solve ik for fixed-base robot with batched IK and collision checking
        init_qpos = robot_helper.init_qpos.to(device)  # chain-sized on a pruned robot
        init_arm_qpos = init_qpos[..., robot_helper.arm_joint_indices]

        num_ik_candidates = cfg.init_num_ik_candidates_per_sample * num_samples_per_scene
        num_scenes = lookat_rot.shape[0]
        batch_size = cfg.init_ik_batch_size
        candidate_lookat_rot, candidate_tl = _batch_sample((lookat_rot, tl), first_mask, num_ik_candidates, device)

        # IK targets are relative to arm_base_link, not the root link (e.g. G1's torso_link is offset).
        arm_base_pose = robot_helper.forward_kinematics_torch(robot_helper.init_qpos[None, :])[
            0, robot_helper.link_names.index(robot_helper.config.arm_base_link_name)
        ]
        world_to_armbase = inverse_tf_mat(arm_base_pose)

        all_arm_qpos = []
        all_valid_mask = []
        per_filter_masks: dict = {}
        for start in range(0, num_ik_candidates, batch_size):
            end = min(start + batch_size, num_ik_candidates)
            batch_lookat_rot = candidate_lookat_rot[:, start:end]
            batch_tl = candidate_tl[:, start:end]

            batch_pose_world = rot_tl_to_tf_mat(rot_mat=batch_lookat_rot, tl=batch_tl).to(device)
            batch_pose_armbase = world_to_armbase @ batch_pose_world
            batch_arm_qpos, batch_ik_success = robot_helper.inverse_kinematics(
                batch_pose_armbase,
                init_qpos=init_arm_qpos,
                position_tolerance=cfg.init_ik_position_tolerance,
                orientation_tolerance=cfg.init_ik_orientation_tolerance,
                use_score_rest=cfg.init_ik_prefer_rest,
            )

            batch_full_qpos = init_qpos.view(1, 1, -1).expand(num_scenes, end - start, -1).clone()
            batch_full_qpos[..., robot_helper.arm_joint_indices] = batch_arm_qpos

            # Filtered per IK batch, not the whole set: collision spheres for num_ik_candidates poses at once won't fit.
            batch_cands = Candidates(
                tl=batch_tl, rot=batch_lookat_rot, qpos=batch_full_qpos, robot=robot_helper, scene_geom=scene_geom
            )
            batch_masks = {f.name: f(batch_cands) for f in self.post_ik_filters}
            batch_masks["ik"] = batch_ik_success.squeeze(-1)

            batch_valid_mask = torch.ones_like(batch_masks["ik"])
            for mask in batch_masks.values():
                batch_valid_mask = batch_valid_mask & mask
            all_arm_qpos.append(batch_arm_qpos)
            all_valid_mask.append(batch_valid_mask)
            for name, mask in batch_masks.items():
                per_filter_masks.setdefault(name, []).append(mask)

            del batch_pose_world, batch_pose_armbase, batch_cands

        arm_qpos = torch.cat(all_arm_qpos, dim=1)
        valid_mask = torch.cat(all_valid_mask, dim=1)
        ik_full_qpos = init_qpos.view(1, 1, -1).expand(num_scenes, arm_qpos.shape[1], -1).clone()
        ik_full_qpos[..., robot_helper.arm_joint_indices] = arm_qpos
        self.diagnostics.update({k: torch.cat(v, dim=1).cpu() for k, v in per_filter_masks.items()})
        self.diagnostics.update(
            {
                "ik_collision": valid_mask.cpu(),
                "ik_tl": candidate_tl.cpu(),
                "ik_qpos": ik_full_qpos.cpu(),
            }
        )
        arm_qpos = _batch_sample(arm_qpos, valid_mask, num_samples_per_scene, device)

        # 8. sample hand joint angles
        full_qpos = init_qpos.view(1, 1, -1).expand(arm_qpos.shape[0], arm_qpos.shape[1], -1).clone()
        full_qpos[..., robot_helper.arm_joint_indices] = arm_qpos

        if cfg.hand_qpos_mu is not None:
            full_qpos[..., robot_helper.hand_joint_indices] = (
                torch.tensor(cfg.hand_qpos_mu, device=device)
                .view(1, 1, -1)
                .expand(full_qpos.shape[0], full_qpos.shape[1], -1)
            )

        hand_qpos_delta = (
            torch.randn_like(full_qpos[..., robot_helper.hand_joint_indices]) * cfg.hand_qpos_jitter_strength
        )
        full_qpos[..., robot_helper.hand_joint_indices] = (
            full_qpos[..., robot_helper.hand_joint_indices] + hand_qpos_delta
        )
        limits = robot_helper.actuated_joint_limits_torch.to(device)
        return full_qpos.clamp(limits[:, 0], limits[:, 1])  # the jitter can leave a joint's range


class FixedBaseSampler(Sampler):
    """Samples initial grasp qpos for a fixed-base robot: look-at poses via collision-free IK."""

    def sample(
        self, scene_geom: SceneGeom, num_samples_per_scene: int
    ) -> Float[torch.Tensor, "n_scene n_samples num_dofs"]:
        cfg = self.cfg
        clst_pts, approach_dir, tl, lookat_rot = self._sample_lookat(
            scene_geom, num_samples_per_scene, cfg.init_num_points_per_sample, center_on_target=False, fold_upper=False
        )

        first_mask = self._run_pre_ik_filters(scene_geom, tl, lookat_rot, clst_pts, approach_dir)
        return self._solve_ik_and_hand(scene_geom, lookat_rot, tl, first_mask, num_samples_per_scene)


class FloatingBaseSampler(Sampler):
    """Samples floating-base variables [tl(3), euler_rxyz(3), hand joints] around the target (no IK)."""

    def sample(
        self, scene_geom: SceneGeom, num_samples_per_scene: int
    ) -> Float[torch.Tensor, "n_scene n_samples num_dofs"]:
        device = self.device
        cfg = self.cfg
        clst_pts, approach_dir, tl, lookat_rot = self._sample_lookat(
            scene_geom, num_samples_per_scene, 10, center_on_target=False, fold_upper=False
        )

        sdf_mask = self._run_pre_ik_filters(scene_geom, tl, lookat_rot, clst_pts, approach_dir)
        lookat_rot, tl = _batch_sample((lookat_rot, tl), sdf_mask, num_samples_per_scene, device)
        euler = matrix_to_euler_angles(lookat_rot, "rxyz")
        qpos_source = cfg.hand_qpos_mu or self.robot_config.init_qpos
        hand_qpos = (
            torch.tensor(qpos_source, device=device).view(1, 1, -1).expand(lookat_rot.shape[0], lookat_rot.shape[1], -1)
        )
        hand_qpos = hand_qpos + torch.randn_like(hand_qpos) * cfg.hand_qpos_jitter_strength
        limits = self.robot_helper.actuated_joint_limits_torch.to(device)
        return torch.cat([tl, euler, hand_qpos.clamp(limits[:, 0], limits[:, 1])], dim=-1)


class WBCSampler(Sampler):
    """Sampler for world-rooted (WBC) arms: approach the target from any azimuth, never from below the table."""

    def sample(
        self, scene_geom: SceneGeom, num_samples_per_scene: int
    ) -> Float[torch.Tensor, "n_scene n_samples num_dofs"]:
        cfg = self.cfg
        clst_pts, approach_dir, tl, lookat_rot = self._sample_lookat(
            scene_geom, num_samples_per_scene, cfg.init_num_points_per_sample, center_on_target=True, fold_upper=True
        )

        first_mask = self._run_pre_ik_filters(scene_geom, tl, lookat_rot, clst_pts, approach_dir)
        return self._solve_ik_and_hand(scene_geom, lookat_rot, tl, first_mask, num_samples_per_scene)


SAMPLER_CLASSES: Dict[str, Type[Sampler]] = {
    "fixed_base": FixedBaseSampler,
    "wbc": WBCSampler,
    "floating_base": FloatingBaseSampler,
}


def build_sampler(
    cfg: GraspOptimConfig,
    robot_config: RobotConfig,
    robot_helper: GraspRobot,
    device: Device,
    sampler_type: Optional[str] = None,
) -> Sampler:
    """Build a sampler; `sampler_type` overrides cfg.sampler_type (fixed_base vs wbc) to force "floating_base"."""
    sampler_type = sampler_type or cfg.sampler_type
    if sampler_type not in SAMPLER_CLASSES:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
    return SAMPLER_CLASSES[sampler_type](cfg, robot_config, robot_helper, device)
