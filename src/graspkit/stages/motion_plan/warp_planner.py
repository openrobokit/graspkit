import dataclasses
import logging
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from robokit.helpers.motion_plan import GradientTrajectoryOptimizerConfig, MotionPlanner, RetimeConfig
from robokit.helpers.motion_plan import MotionPlanContext as ctx
from robokit.helpers.motion_plan import MotionPlannerConfig as PlanConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.robo.robot import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.trajectory_task import TrajectoryTask
from torch.types import Device

from graspkit.configs.experiment_config import MotionPlannerConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.motion_plan.torch_planner import TorchPlanner

logger = logging.getLogger("graspkit")

# robokit 48d3d15's `offline_joint_lbfgs` preset, kept here so robokit's retuning does not move grasp trajectories;
# normalized directions: raw collision gradients overshoot every fixed step, which left straight-line paths in collision.
JOINT_GOAL_PLAN = PlanConfig(
    stages=(
        GradientTrajectoryOptimizerConfig(
            terms=[
                TrajectoryCollisionTask(
                    scene=ctx.SCENE, scene_indices=ctx.SCENE_INDICES, weight=1000.0, margin=0.025, name="collision"
                ),
                TrajectorySmoothnessTask(weight=200.0, smooth_boundary=False, name="smoothness"),
                TrajectorySmoothnessTask(
                    weight=30.0, order=2, dt=ctx.STEP, history=ctx.HISTORY, name="acceleration_smoothness"
                ),
                TrajectoryLimitTask(
                    order=1,
                    upper=ctx.VELOCITY_LIMITS,
                    weight=5.0,
                    dt=ctx.STEP,
                    stencil="forward",
                    name="velocity_limit",
                ),
                TrajectoryTask(PositionLimit(weight=100.0), name="position_limit"),
                TrajectoryTask(ConfigurationTask(rest_q="midrange", weight=0.01), name="rest"),
            ],
            solver=PopulationSolverConfig(
                num_iters=100,
                num_candidates=[8],
                # normalized: trial steps are joint-space lengths; raw first steps along the 1e6-scale collision
                # gradient failed every Armijo trial, so the plan never left its linear seed
                optimizer=LBFGSOptimizerConfig(
                    line_search_alphas=(0.001, 0.01, 0.05, 0.1),
                    history_len=15,
                    fold_line_search_gradient=True,
                    normalize_direction=True,
                ),
            ),
            lock_frames=(0, -1),
        ),
    ),
    jerk_limit=100.0,
    retime=RetimeConfig(speed_scale=1.0, max_dt=1.0),
)


class WarpPlanner(TorchPlanner):
    """TorchPlanner on robokit's MotionPlanner: joint-goal plans, or one plan from start through the lift."""

    def __init__(self, cfg: MotionPlannerConfig, robot_config: RobotConfig, device: Device = "cpu"):
        super().__init__(cfg, robot_config, device=device)
        self._device_str = (
            str(device)
            if not isinstance(device, torch.device)
            else (f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu")
        )
        # Active DOFs = arm + hand minus start_from_grasp_dof_indices; a zero mask entry pins the joint.
        frozen = set(self.cfg.start_from_grasp_dof_indices or ())
        active = torch.cat([self.robot_helper.arm_joint_indices, self.robot_helper.hand_joint_indices])
        self._active_joint_mask = np.zeros(self.robot_helper.num_actuated_joints, dtype=np.float32)
        self._active_joint_mask[[int(i) for i in active if int(i) not in frozen]] = 1.0
        # robot_helper lacks robokit spheres; Robot.load drops keep_world_link/ee_link_names, so parse directly.
        spec = RobotSpec.parse(
            robot_description_or_path=self.robot_config.urdf_or_mjcf_path,
            load_collision_spheres=True,
            collision_spheres_path=self.robot_config.collision_spheres_path,
            self_collision_ignore_path=self.robot_config.get_self_collision_ignore_path("sphere"),
            mesh_dir=self.robot_config.mesh_dir,
            keep_mjcf_world_link=self.robot_config.keep_world_link,
            ee_link_names=self.robot_config.ee_link_names,
        )
        for name, limit in (self.robot_config.joint_velocity_limits or {}).items():
            spec.actuated_joint_velocity_limits[spec.actuated_joint_names.index(name)] = limit
        self._warp_robot = Robot(spec=spec)
        # Planner fixes frame count at construction; optimize at num_opt_frames, interpolated downstream.
        self._num_opt_frames = min(32, self.cfg.num_steps_pre_grasp)
        # one candidate: joint goals skip IK and our linear init is good; extra candidates just multiply memory and OOM
        stage = JOINT_GOAL_PLAN.stages[0]
        stage = dataclasses.replace(
            stage,
            terms=deepcopy(stage.terms),
            solver=dataclasses.replace(stage.solver, num_iters=self.cfg.traj_opt_max_iter, num_candidates=[1]),
        )
        stage.task("smoothness").weight = self.cfg.traj_opt_smoothness_weight
        stage.task("acceleration_smoothness").weight = self.cfg.traj_opt_accel_smoothness_weight
        if self.cfg.traj_opt_self_collision_weight > 0:
            stage.add(
                TrajectorySelfCollisionTask(
                    weight=self.cfg.traj_opt_self_collision_weight, margin=self.cfg.traj_opt_self_collision_margin
                )
            )
        preset = dataclasses.replace(JOINT_GOAL_PLAN, num_timesteps=self._num_opt_frames - 1, stages=(stage,))
        self._preset = preset
        # Floating hands set no arm chain / hand_base_link_name; the planner's ee link is then the palm.
        self._ee_link = self.robot_config.hand_base_link_name or self.robot_config.palm_link_name

    @torch.no_grad()
    def plan_full(
        self,
        start_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        pre_grasp_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        grasp_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        post_grasp_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        lift_qpos: Optional[Float[torch.Tensor, "n_scenes n_grasps num_dofs"]],
        scene_geom: SceneGeom,
    ) -> Tuple[Float[torch.Tensor, "n_scenes n_grasps num_steps num_dofs"], Float[torch.Tensor, "n_scenes n_grasps"]]:
        """One robokit plan: start → pre-grasp against the scene, then grasp, the locked hold and the lift without it.

        robokit times every segment but the hold, which keeps num_steps_holding; the plan is sampled every control_dt
        and padded with its last frame to num_steps when shorter. Returns it and the start → pre-grasp penetration.
        """
        n_scenes, n_trajs = grasp_qpos.shape[:2]
        hand = self.robot_helper.hand_joint_indices
        # the hand closes over the hold; or, as the phased plan does, at once after the grasp, applied once planned
        closed = post_grasp_qpos if self.cfg.close_during_holding else grasp_qpos
        lifts = []
        if lift_qpos is not None:
            lifts = [lift_qpos.clone()]
            lifts[0][..., hand] = closed[..., hand]
        flat = [rearrange(q, "s t d -> (s t) d") for q in (start_qpos, pre_grasp_qpos, grasp_qpos, closed, *lifts)]
        hold = self.cfg.num_steps_holding * self.cfg.control_dt
        # start → pre-grasp, the approach and the lift share frames by the time each needs; the hold keeps its own
        steps = 31 + max(2, round(hold / self._preset.dt))
        scene = scene_geom.build_collision_scene_in_base(self.robot_pose_mat.cpu().numpy())
        if self.cfg.traj_opt_self_collision_weight > 0:
            self.robot_config.get_self_collision_ignore_path("sphere", warn=True)
        planner = MotionPlanner(
            dataclasses.replace(self._preset, num_timesteps=steps),
            self._warp_robot,
            self._ee_link,
            scene=scene,
            device=self._device_str,
        )
        planner.set_active_joint_mask(self._active_joint_mask)
        result = planner.solve_offline_numpy(
            [q.cpu().numpy().astype(np.float32) for q in flat],
            scene_indices=np.repeat(np.arange(n_scenes, dtype=np.int32), n_trajs),
            segment_check_collision=(True, False, False, *(False,) * len(lifts)),
            segment_fixed_duration=(None, None, hold, *(None,) * len(lifts)),
            segment_locked=(False, False, True, *(False,) * len(lifts)),
        )
        frames = rearrange(torch.from_numpy(result.q_traj), "(s t) f d -> s t f d", s=n_scenes, t=n_trajs)
        max_pene = self.compute_energy(
            frames[..., : result.waypoint_indices[1] + 1, :].to(grasp_qpos.device), scene_geom
        )
        trajs = torch.from_numpy(result.interpolate(self.cfg.control_dt)).to(grasp_qpos.device)
        if not self.cfg.close_during_holding:
            reached = (trajs - flat[2][:, None]).abs().amax(dim=-1).lt(1e-6).int().argmax(dim=-1)  # the grasp sample
            after = torch.arange(trajs.shape[1], device=trajs.device) > reached[:, None]
            post = rearrange(post_grasp_qpos, "s t d -> (s t) d")[:, None, hand]
            trajs[..., hand] = torch.where(after[..., None], post, trajs[..., hand])
        trajs = torch.cat([trajs, trajs[:, -1:].expand(-1, max(self.cfg.num_steps - trajs.shape[1], 0), -1)], dim=1)
        return rearrange(trajs, "(s t) f d -> s t f d", s=n_scenes, t=n_trajs), max_pene["max_pene"]

    @torch.no_grad()
    def optimize_trajs(
        self,
        init_trajs: Float[torch.Tensor, "... num_steps num_dofs"],
        scene_geom: SceneGeom,
    ) -> Tuple[Float[torch.Tensor, "... num_steps num_dofs"], Float[torch.Tensor, "..."]]:
        n_scenes, n_trajs = init_trajs.shape[:2]
        # robokit FK is base-relative; remap the scene into the robot base frame (single scene in).
        scene = scene_geom.build_collision_scene_in_base(self.robot_pose_mat.cpu().numpy())
        start_q = rearrange(init_trajs[..., 0, :], "s t d -> (s t) d").cpu().numpy().astype(np.float32)
        goal_q = rearrange(init_trajs[..., -1, :], "s t d -> (s t) d").cpu().numpy().astype(np.float32)
        scene_index_by_query = np.repeat(np.arange(n_scenes, dtype=np.int32), n_trajs)

        # MotionPlanner fixes its scene at construction (robokit 234ba00e), so build a fresh planner per call.
        if self.cfg.traj_opt_self_collision_weight > 0:
            self.robot_config.get_self_collision_ignore_path("sphere", warn=True)
        planner = MotionPlanner(self._preset, self._warp_robot, self._ee_link, scene=scene, device=self._device_str)
        planner.set_active_joint_mask(self._active_joint_mask)
        result = planner.solve_offline_numpy([start_q, goal_q], scene_indices=scene_index_by_query)
        opt_trajs = rearrange(
            torch.from_numpy(result.q_traj).to(init_trajs.device), "(s t) f d -> s t f d", s=n_scenes, t=n_trajs
        )
        # Reuse the base class's penetration recipe (same sign convention as torch/warp_adam).
        max_pene = self.compute_energy(opt_trajs, scene_geom)["max_pene"]
        return opt_trajs, max_pene
