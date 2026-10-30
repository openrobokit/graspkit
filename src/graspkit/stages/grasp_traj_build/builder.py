"""GraspTrajBuilder: grasp poses -> deployable trajectories, optionally converting floating to fixed-base first."""

import dataclasses
import logging
from typing import List, Optional, Tuple

import torch
from jaxtyping import Bool, Float
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix
from torch.types import Device

from graspkit.configs.experiment_config import GraspTrajBuilderConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import get_table_only_scene_descriptions
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.grasp_filters import (
    Candidates,
    ContactAvoid,
    Filter,
    Ori,
    Pen,
    RestPose,
)
from graspkit.stages.grasp_traj_build.ik_convert import ik_convert_floating_to_fixed_batch
from graspkit.stages.motion_plan import build_motion_planner

logger = logging.getLogger("graspkit")


class GraspTrajBuilder:
    """Convert floating-base grasps into deployable trajectories: filter, optional IK-convert to fixed, then plan."""

    def __init__(self, cfg: GraspTrajBuilderConfig, device: Device):
        if cfg.convert_to_fixed:
            assert cfg.fixed_base_robot_config.base_type == "fixed", (
                "floating->fixed conversion requires a fixed-base robot config"
            )
        self.cfg = cfg
        robot_config = cfg.fixed_base_robot_config if cfg.convert_to_fixed else cfg.floating_base_robot_config
        self.robot = GraspRobot.from_config(robot_config, device=device)
        # filters run before IK conversion, so their FK must use the robot input grasp_qpos belongs to
        self.filter_robot = (
            GraspRobot.from_config(cfg.floating_base_robot_config, device=device)
            if cfg.convert_to_fixed
            else self.robot
        )
        self.motion_planner = build_motion_planner(cfg.motion_planner, robot_config, device=device)
        # built once so callers (optimize_grasps.py --verbose) can read which filters run without re-deriving flags
        self.filters: List[Filter] = []
        if cfg.contact_avoid_filter:
            self.filters.append(ContactAvoid())
        if cfg.pen_filter:
            self.filters.append(Pen(cfg.pen_filter_threshold))
        if cfg.rest_pose_filter_max_dist is not None:
            self.filters.append(RestPose(cfg.rest_pose_filter_max_dist))
        if cfg.hand_axis is not None:
            self.filters.append(Ori(cfg.hand_axis, cfg.hand_axis_target, cfg.filter_hand_axis_max_angle_deg))

    def build(
        self,
        batch: GraspKitData,
        start_qpos: Optional[Float[torch.Tensor, "..."]] = None,  # noqa: F821
        skip_filter: bool = False,
        target_xy: Optional[torch.Tensor] = None,
    ) -> Tuple[GraspKitData, Bool[torch.Tensor, "B K"]]:
        """`target_xy` (S, 2) overrides the ori-gate target when scenes lack a target mesh (e.g. deploy perception)."""
        cfg = self.cfg
        num_scenes, num_grasps = batch.grasp_qpos.shape[:2]
        success = torch.ones((num_scenes, num_grasps), dtype=torch.bool, device=self.robot.device)
        metrics: List[str] = []
        if not skip_filter:
            success, metrics = self.filter_grasps(batch, success)

        if cfg.convert_to_fixed:
            GraspRobot._get_ik_helper.cache_clear()
            converted, ik_ok = ik_convert_floating_to_fixed_batch(
                batch.to_packed(),
                cfg.floating_base_robot_config,
                self.robot,
                cfg.arm_range_fraction,
                target_xy,
                cfg.ori_threshold_deg,
            )
            success = success & ik_ok.view(num_scenes, num_grasps)
            batch = converted.to_padded()
        detail = f" ({', '.join(metrics)})" if metrics else ""
        logger.info(f"traj builder: {int(success.sum())}/{success.numel()} grasps pass{detail}")

        if batch.pre_grasp_qpos is None or batch.post_grasp_qpos is None:
            batch = self.motion_planner.plan(batch, return_pre_post_only=True)

        scene_descriptions = batch.scene_descriptions
        if cfg.plan_table_only and "table" in scene_descriptions[0].boxes:
            table_only = [
                get_table_only_scene_descriptions(
                    table_size=tuple(scene.boxes["table"].size),
                    table_pos=scene.boxes["table"].position,
                    table_quat=scene.boxes["table"].quaternion,
                )
                for scene in scene_descriptions
            ]
            batch = dataclasses.replace(
                batch,
                scene_descriptions=table_only,
                scene_ids=[scene.scene_id for scene in table_only],
            )
        planned = self.motion_planner.plan(batch, start_qpos=start_qpos, return_pre_post_only=False)
        planned.scene_descriptions = scene_descriptions
        planned.scene_ids = [d.scene_id for d in scene_descriptions]
        return planned, success

    def filter_grasps(
        self, batch: GraspKitData, success: Bool[torch.Tensor, "B K"]
    ) -> Tuple[Bool[torch.Tensor, "B K"], List[str]]:
        # Per-filter masks of this call, kept for optimize_grasps.py --verbose (success AND-s them away).
        self.last_masks = {}
        if self.filters:
            cands = self._build_candidates(batch)  # one SceneGeom + FK shared by every filter
            self.last_masks = {f.name: f(cands) for f in self.filters}
        metrics = [f"{name}={mask.float().mean():.2%}" for name, mask in self.last_masks.items()]
        for mask in self.last_masks.values():
            success = success & mask.to(success.device)
        return success, metrics

    def _build_candidates(self, batch: GraspKitData) -> Candidates:
        """Optimized grasps as filter input. tl/rot: the floating root pose, else FK on the hand base link."""
        qpos = batch.grasp_qpos.to(self.robot.device)
        robot = self.filter_robot
        if robot.config.base_type == "floating":  # floating hands have no hand_base_link_name
            tl, rot = qpos[..., :3], euler_angles_to_matrix(qpos[..., 3:6], "rxyz")
        else:
            link_poses = robot.forward_kinematics_torch(qpos)
            hand_pose = link_poses[..., robot.link_names.index(robot.config.hand_base_link_name), :, :]
            tl, rot = hand_pose[..., :3, 3], hand_pose[..., :3, :3]
        return Candidates(
            tl=tl,
            rot=rot,
            qpos=qpos,
            robot=robot,
            scene_geom=SceneGeom(batch.scene_descriptions, device=self.robot.device),
        )
