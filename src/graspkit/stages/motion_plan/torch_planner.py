import dataclasses
import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from robokit.xform.warp.torch_wrappers import inverse_tf_mat
from torch.types import Device

from graspkit.configs.experiment_config import MotionPlannerConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_geom import SceneGeom
from graspkit.utils.tui_utils import log_once

logger = logging.getLogger("graspkit")


class TorchPlanner:
    def __init__(self, cfg: MotionPlannerConfig, robot_config: RobotConfig, device: Device = "cpu"):
        self.cfg = cfg
        self.device = device

        self.robot_config = robot_config
        self.robot_helper = GraspRobot.from_config(self.robot_config, device=self.device)
        self.robot_pose_mat = self.robot_helper.root_pose_mat
        self.robot_init_qpos = self.robot_helper.init_qpos
        logger.info(
            f"TorchPlanner robot: {len(self.robot_helper.link_names)} links, "
            f"{self.robot_helper.num_actuated_joints} DOFs"
            + (" (pruned chain)" if self.robot_config.ee_link_names is not None else "")
        )

    @torch.no_grad()
    def plan(
        self,
        batch: GraspKitData,
        start_qpos: Optional[Float[torch.Tensor, "num_dofs"]] = None,  # noqa: F821
        return_pre_post_only: bool = False,
    ) -> GraspKitData:
        """Plan a padded batch: needs grasp_qpos, returns pre/post-grasp qpos + full_traj_qpos unless pre-post-only."""
        scene_descriptions = batch.scene_descriptions
        grasp_qpos = batch.grasp_qpos
        caller_pre_grasp_qpos = batch.pre_grasp_qpos
        caller_post_grasp_qpos = batch.post_grasp_qpos
        device = grasp_qpos.device
        num_scenes = len(scene_descriptions)
        num_samples_per_scene = grasp_qpos.shape[1]
        num_dofs = self.robot_helper.num_actuated_joints
        if self.robot_config.base_type == "floating":
            num_dofs += 6
        num_batches = math.ceil(num_scenes * num_samples_per_scene / self.cfg.num_samples_per_batch)
        num_samples_per_scene_per_batch = math.ceil(num_samples_per_scene / num_batches)
        batch_remainder = (num_scenes * num_samples_per_scene) % self.cfg.num_samples_per_batch
        if num_batches > 1 and batch_remainder != 0 and batch_remainder < 128:
            log_once(
                f"Performance warning: Last batch contains only {batch_remainder} samples (< 128). "
                f"For optimal GPU utilization, adjust num_scenes ({num_scenes}) and num_samples_per_scene ({num_samples_per_scene}) "
                f"so their product is divisible by num_samples_per_batch ({self.cfg.num_samples_per_batch}).",
                level=logging.WARNING,
                logger=logger,
            )

        start_qpos = start_qpos if start_qpos is not None else self.robot_init_qpos
        frozen_dof_indices = self.cfg.start_from_grasp_dof_indices
        if frozen_dof_indices is not None:
            start_qpos = start_qpos.broadcast_to(grasp_qpos.shape).clone()
            for idx in frozen_dof_indices:
                start_qpos[..., idx] = grasp_qpos[..., idx]

        scene_geom = SceneGeom(scene_descriptions, device=self.device)
        pre_grasp_qpos = torch.zeros((num_scenes, num_samples_per_scene, num_dofs), device=device)
        post_grasp_qpos = torch.zeros((num_scenes, num_samples_per_scene, num_dofs), device=device)
        if not return_pre_post_only:
            full_trajs = torch.zeros((num_scenes, num_samples_per_scene, self.cfg.num_steps, num_dofs), device=device)
            pre_trajs_pene = torch.ones((num_scenes, num_samples_per_scene), device=device) * -1.0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * num_samples_per_scene_per_batch
            end_idx = min(start_idx + num_samples_per_scene_per_batch, num_samples_per_scene)
            batch_grasp_qpos = grasp_qpos[:, start_idx:end_idx, :]
            if caller_pre_grasp_qpos is None or caller_post_grasp_qpos is None:
                batch_pre_grasp_qpos, batch_post_grasp_qpos = self.get_pre_post_grasp_qpos(
                    scene_geom=scene_geom, grasp_qpos=batch_grasp_qpos
                )
            else:
                batch_pre_grasp_qpos = caller_pre_grasp_qpos[:, start_idx:end_idx, :]
                batch_post_grasp_qpos = caller_post_grasp_qpos[:, start_idx:end_idx, :]

            if frozen_dof_indices is not None:
                for idx in frozen_dof_indices:
                    batch_pre_grasp_qpos[..., idx] = batch_grasp_qpos[..., idx]
                    batch_post_grasp_qpos[..., idx] = batch_grasp_qpos[..., idx]

            pre_grasp_qpos[:, start_idx:end_idx, :] = batch_pre_grasp_qpos
            post_grasp_qpos[:, start_idx:end_idx, :] = batch_post_grasp_qpos
            if return_pre_post_only:
                continue

            batch_start_qpos = (
                start_qpos[:, start_idx:end_idx, :] if start_qpos.ndim == batch_pre_grasp_qpos.ndim else start_qpos
            )
            batch_target_qpos = None
            if self.cfg.num_steps_post_grasp > 0:
                batch_target_qpos = self.get_lifting_target_qpos(post_grasp_qpos=batch_post_grasp_qpos)
                if frozen_dof_indices is not None:
                    for idx in frozen_dof_indices:
                        batch_target_qpos[..., idx] = batch_grasp_qpos[..., idx]
            if self.robot_config.base_type == "fixed" and self.cfg.enable_joint_optimization:
                batch_full_trajs, pre_trajs_pene[:, start_idx:end_idx] = self.plan_full(
                    batch_start_qpos.broadcast_to(batch_grasp_qpos.shape),
                    batch_pre_grasp_qpos,
                    batch_grasp_qpos,
                    batch_post_grasp_qpos,
                    batch_target_qpos,
                    scene_geom,
                )
                # a longer plan lengthens every trajectory, each holding its last frame
                pad = batch_full_trajs.shape[-2] - full_trajs.shape[-2]
                full_trajs = torch.cat([full_trajs, full_trajs[..., -1:, :].expand(-1, -1, max(pad, 0), -1)], dim=-2)
                batch_full_trajs = torch.cat(
                    [batch_full_trajs, batch_full_trajs[..., -1:, :].expand(-1, -1, max(-pad, 0), -1)], dim=-2
                )
                full_trajs[:, start_idx:end_idx] = batch_full_trajs
                continue
            if self.robot_config.base_type == "fixed":
                batch_pre_trajs = self.interpolate_qpos(batch_start_qpos, batch_pre_grasp_qpos, num_steps=50)
                batch_pre_trajs, batch_pre_trajs_pene = self.optimize_trajs(
                    init_trajs=batch_pre_trajs, scene_geom=scene_geom
                )
                if frozen_dof_indices is not None:
                    for idx in frozen_dof_indices:
                        batch_pre_trajs[..., idx] = batch_grasp_qpos[..., idx].unsqueeze(-1)
                batch_pre_trajs = self.interpolate_trajs(batch_pre_trajs, self.cfg.num_steps_pre_grasp)
                pre_trajs_pene[:, start_idx:end_idx] = batch_pre_trajs_pene
            else:
                batch_pre_grasp_pene = self.compute_pose_penetration(qpos=batch_pre_grasp_qpos, scene_geom=scene_geom)
                pre_trajs_pene[:, start_idx:end_idx] = batch_pre_grasp_pene

            if batch_target_qpos is not None:
                batch_post_trajs = self.interpolate_qpos(
                    batch_post_grasp_qpos, batch_target_qpos, num_steps=10
                )  # no need to optimize
                batch_post_trajs = self.interpolate_trajs(batch_post_trajs, self.cfg.num_steps_post_grasp)
            batch_pre_to_grasp_trajs = self.interpolate_qpos(
                batch_pre_grasp_qpos, batch_grasp_qpos, self.cfg.num_steps_pre_to_grasp
            )
            batch_holding_trajs = self.interpolate_qpos(
                batch_grasp_qpos if self.cfg.close_during_holding else batch_post_grasp_qpos,
                batch_post_grasp_qpos,
                self.cfg.num_steps_holding,
            )
            if self.robot_config.base_type == "fixed":
                parts = [batch_pre_trajs, batch_pre_to_grasp_trajs, batch_holding_trajs]
                if self.cfg.num_steps_post_grasp > 0:
                    parts.append(batch_post_trajs)
                batch_full_trajs = torch.cat(parts, dim=-2)
            else:
                parts = []
                if self.cfg.num_steps_pre_grasp > 0:
                    batch_pre_trajs = self.interpolate_qpos(
                        batch_pre_grasp_qpos, batch_pre_grasp_qpos, self.cfg.num_steps_pre_grasp
                    )
                    parts.append(batch_pre_trajs)
                parts.append(batch_pre_to_grasp_trajs)
                parts.append(batch_holding_trajs)
                if self.cfg.num_steps_post_grasp > 0:
                    parts.append(batch_post_trajs)
                batch_full_trajs = torch.cat(parts, dim=-2)

            full_trajs[:, start_idx:end_idx, :, :] = batch_full_trajs

        if return_pre_post_only:
            return dataclasses.replace(batch, pre_grasp_qpos=pre_grasp_qpos, post_grasp_qpos=post_grasp_qpos)
        return dataclasses.replace(
            batch,
            pre_grasp_qpos=pre_grasp_qpos,
            post_grasp_qpos=post_grasp_qpos,
            full_traj_qpos=full_trajs,
            pre_traj_penetration=pre_trajs_pene,
        )

    def plan_full(
        self,
        start_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        pre_grasp_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        grasp_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        post_grasp_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        lift_qpos: Optional[Float[torch.Tensor, "n_scenes n_grasps num_dofs"]],
        scene_geom: SceneGeom,
    ) -> Tuple[Float[torch.Tensor, "n_scenes n_grasps num_steps num_dofs"], Float[torch.Tensor, "n_scenes n_grasps"]]:
        """One plan through start, pre-grasp, grasp, hold and lift (`enable_joint_optimization`), at least num_steps
        long; the warp backend's."""
        raise NotImplementedError("enable_joint_optimization needs the warp backend.")

    @torch.enable_grad()
    def get_pre_post_grasp_qpos(
        self,
        scene_geom: SceneGeom,
        grasp_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
    ) -> Tuple[
        Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
        Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
    ]:
        grasp_qpos = grasp_qpos.detach().clone()
        grasp_qpos.requires_grad_(True)
        grasp_link_poses = self.robot_helper.forward_kinematics_torch(grasp_qpos)
        spheres = self.robot_helper.get_collision_spheres(link_poses=grasp_link_poses)
        spheres = rearrange(spheres, "n_scenes n_grasps n_spheres dim -> n_scenes (n_grasps n_spheres) dim")
        center_sdf, _, _ = scene_geom.query_target_object_signed_distances(spheres[..., :3])  # outside is positive
        sphere_sdf = center_sdf - spheres[..., 3]  # outside is positive
        sphere_sdf.sum().backward()  # compute gradients to move spheres away from the object

        if self.robot_config.base_type == "floating":
            pregrasp_qpos = grasp_qpos.clone()
            pregrasp_qpos[..., :3] += self.cfg.pre_grasp_arm_grad_move * F.normalize(grasp_qpos.grad[..., :3], dim=-1)
            pregrasp_qpos[..., 6:] += self.cfg.pre_grasp_hand_grad_move * grasp_qpos.grad[..., 6:]
            postgrasp_qpos = grasp_qpos.clone()
            postgrasp_qpos[..., 6:] -= self.cfg.post_grasp_hand_grad_move * grasp_qpos.grad[..., 6:]
        else:
            pregrasp_qpos = grasp_qpos.clone()
            if self.cfg.pre_grasp_arm_grad_move > 0:
                pregrasp_qpos[..., self.robot_helper.arm_joint_indices] += (
                    self.cfg.pre_grasp_arm_grad_move * grasp_qpos.grad[..., self.robot_helper.arm_joint_indices]
                )  # move arm away from the object
                pregrasp_qpos[..., self.robot_helper.hand_joint_indices] = self.robot_init_qpos[
                    self.robot_helper.hand_joint_indices
                ]
            else:
                if self.cfg.pre_grasp_hand_grad_move <= 0:
                    raise ValueError("pre_grasp_hand_grad_move must be > 0 when pre_grasp_arm_grad_move is 0")
                pregrasp_qpos[..., self.robot_helper.hand_joint_indices] += (
                    self.cfg.pre_grasp_hand_grad_move * grasp_qpos.grad[..., self.robot_helper.hand_joint_indices]
                )  # move hand into the object
            postgrasp_qpos = grasp_qpos.clone()
            postgrasp_qpos[..., self.robot_helper.hand_joint_indices] -= (
                self.cfg.post_grasp_hand_grad_move * grasp_qpos.grad[..., self.robot_helper.hand_joint_indices]
            )  # move hand into the object

        joint_limits = self.robot_helper.spec.get_tensors_torch(device=self.device).actuated_joint_limits
        if joint_limits is not None:
            if self.robot_config.base_type == "floating":
                pregrasp_qpos[..., 6:] = torch.clamp(pregrasp_qpos[..., 6:], joint_limits[:, 0], joint_limits[:, 1])
                postgrasp_qpos[..., 6:] = torch.clamp(postgrasp_qpos[..., 6:], joint_limits[:, 0], joint_limits[:, 1])
            else:
                pregrasp_qpos = torch.clamp(pregrasp_qpos, joint_limits[:, 0], joint_limits[:, 1])
                postgrasp_qpos = torch.clamp(postgrasp_qpos, joint_limits[:, 0], joint_limits[:, 1])

        return pregrasp_qpos.detach(), postgrasp_qpos.detach()

    @torch.no_grad()
    def interpolate_qpos(
        self,
        init_qpos: Float[torch.Tensor, "... num_dofs"],
        end_qpos: Float[torch.Tensor, "... num_dofs"],
        num_steps: int,
    ) -> Float[torch.Tensor, "... num_steps num_dofs"]:
        init_qpos, end_qpos = torch.broadcast_tensors(init_qpos, end_qpos)
        steps = torch.linspace(0, 1, num_steps, device=init_qpos.device).unsqueeze(-1)
        return init_qpos.unsqueeze(-2) + steps * (end_qpos - init_qpos).unsqueeze(-2)

    # TODO why we have this
    @torch.no_grad()
    def compute_pose_penetration(
        self,
        qpos: Float[torch.Tensor, "... num_dofs"],
        scene_geom: SceneGeom,
    ) -> Float[torch.Tensor, "..."]:
        link_poses = self.robot_helper.forward_kinematics_torch(qpos)
        spheres = self.robot_helper.get_collision_spheres(link_poses=link_poses)
        spheres = rearrange(spheres, "n_scenes n_grasps n_spheres dim -> n_scenes (n_grasps n_spheres) dim")
        center_sdf, _, _ = scene_geom.query_scene_signed_distances(spheres[..., :3])  # outside is positive
        sphere_sdf = center_sdf - spheres[..., 3]
        sphere_sdf = -sphere_sdf  # flip sign, now inside is positive
        sphere_sdf += 0.002  # add a small offset to avoid penetration
        sphere_sdf[sphere_sdf <= 0] = 0  # only consider penetration
        sphere_sdf = rearrange(
            sphere_sdf,
            "n_scenes (n_grasps n_spheres) -> n_scenes n_grasps (n_spheres)",
            n_grasps=qpos.shape[1],
        )
        return sphere_sdf.max(dim=-1).values

    @torch.no_grad()
    def interpolate_trajs(
        self,
        orig_trajs: Float[torch.Tensor, "n_scenes n_trajs n_steps num_dofs"],
        num_steps: int,
    ) -> Float[torch.Tensor, "n_scenes n_trajs num_steps num_dofs"]:
        interp_trajs = F.interpolate(
            rearrange(
                orig_trajs,
                "n_scenes n_trajs n_steps num_dofs -> (n_scenes n_trajs) num_dofs n_steps",
            ),
            size=num_steps,
            mode="linear",
            align_corners=True,
        )
        return rearrange(
            interp_trajs,
            "(n_scenes n_trajs) num_dofs n_steps -> n_scenes n_trajs n_steps num_dofs",
            n_scenes=orig_trajs.shape[0],
            n_trajs=orig_trajs.shape[1],
        )

    @torch.no_grad()
    def compute_energy(
        self,
        full_traj_qpos: Float[torch.Tensor, "n_scenes n_trajs n_steps num_dofs"],
        scene_geom: SceneGeom,
    ) -> dict[str, torch.Tensor]:
        """Compute E_pen + E_smooth for a trajectory; mirrors `optimize_trajs` loss, a benchmark reference metric."""
        helper = self.robot_helper
        traj = full_traj_qpos
        n_scenes, n_trajs, n_frame, _ = traj.shape

        link_poses = helper.forward_kinematics_torch(traj)
        spheres = helper.get_collision_spheres(link_poses=link_poses)
        spheres = rearrange(spheres, "s t f n d -> s (t f n) d")
        center_sdf, _, _ = scene_geom.query_scene_signed_distances(spheres[..., :3])
        sphere_sdf = -(center_sdf - spheres[..., 3]) + self.cfg.padding_dist
        sphere_sdf[sphere_sdf <= 0] = 0
        sphere_sdf = rearrange(sphere_sdf, "s (t f n) -> s t (f n)", t=n_trajs, f=n_frame)
        max_pene = sphere_sdf.max(dim=-1).values - self.cfg.padding_dist
        E_pen = F.mse_loss(sphere_sdf.sum(dim=-1), torch.zeros(n_scenes, n_trajs, device=traj.device))
        E_pen = E_pen * self.cfg.pen_weight

        joint_weights = torch.zeros(traj.shape[-1], device=traj.device)
        joint_weights[helper.arm_joint_indices] = self.cfg.arm_smooth_weight
        joint_weights[helper.hand_joint_indices] = self.cfg.hand_smooth_weight
        dis_diff_std = torch.diff(traj, dim=-2).std(dim=-2)
        vel_diff_std = torch.diff(torch.diff(traj, dim=-2), dim=-2).std(dim=-2)
        smooth_term = ((dis_diff_std + vel_diff_std * self.cfg.vel_smooth_weight) * joint_weights).sum(dim=-1)
        E_smooth = F.mse_loss(smooth_term, torch.zeros_like(smooth_term))

        return {"E_pen": E_pen, "E_smooth": E_smooth, "max_pene": max_pene}

    @torch.enable_grad()
    def optimize_trajs(
        self,
        init_trajs: Float[torch.Tensor, "... num_steps num_dofs"],
        scene_geom: SceneGeom,
    ) -> Tuple[Float[torch.Tensor, "... num_steps num_dofs"], Float[torch.Tensor, "..."]]:
        helper = self.robot_helper

        start_qpos = init_trajs[..., 0:1, :].detach().clone()
        end_qpos = init_trajs[..., -2:-1, :].detach().clone()
        opt_qpos = init_trajs[..., 1:-1, :].detach().clone()

        opt_qpos.requires_grad_(True)
        optimizer = torch.optim.AdamW([opt_qpos], lr=0.02, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.opt_steps, eta_min=5e-5)

        best_loss = float("inf")
        best_opt_qpos = opt_qpos.clone()
        best_max_pene = None

        joints_smooth_weights = torch.zeros(opt_qpos.shape[-1], device=opt_qpos.device)
        joints_smooth_weights[helper.arm_joint_indices] = self.cfg.arm_smooth_weight
        joints_smooth_weights[helper.hand_joint_indices] = self.cfg.hand_smooth_weight

        for step in range(self.cfg.opt_steps + 1):
            full_traj_qpos = torch.cat([start_qpos, opt_qpos, end_qpos], dim=-2)
            link_poses = helper.forward_kinematics_torch(full_traj_qpos)
            spheres = helper.get_collision_spheres(link_poses=link_poses)

            spheres = rearrange(
                spheres,
                "n_scenes n_trajs n_frame n_sphere dim -> n_scenes (n_trajs n_frame n_sphere) dim",
            )
            center_sdf, _, _ = scene_geom.query_scene_signed_distances(spheres[..., :3])
            sphere_sdf = center_sdf - spheres[..., 3]
            sphere_sdf = -sphere_sdf
            sphere_sdf += self.cfg.padding_dist
            sphere_sdf[sphere_sdf <= 0] = 0
            sphere_sdf = rearrange(
                sphere_sdf,
                "n_scenes (n_trajs n_frame n_sphere) -> n_scenes n_trajs (n_frame n_sphere)",
                n_trajs=full_traj_qpos.shape[1],
                n_frame=full_traj_qpos.shape[2],
            )
            max_pene = sphere_sdf.max(dim=-1).values - self.cfg.padding_dist
            sphere_sdf = sphere_sdf.sum(dim=-1)
            E_pen = F.mse_loss(sphere_sdf, torch.zeros_like(sphere_sdf)) * self.cfg.pen_weight

            dis_diff = torch.diff(full_traj_qpos, dim=-2)
            dis_diff_std = dis_diff.std(dim=-2)
            vel_diff = torch.diff(dis_diff, dim=-2)
            vel_diff_std = vel_diff.std(dim=-2)
            E_smooth = dis_diff_std + vel_diff_std * self.cfg.vel_smooth_weight
            E_smooth = E_smooth * joints_smooth_weights
            E_smooth = E_smooth.sum(dim=-1)
            E_smooth = F.mse_loss(E_smooth, torch.zeros_like(E_smooth))

            cost = E_pen + E_smooth

            if step % 100 == 0:
                logger.info(
                    f"Step {step}/{self.cfg.opt_steps}, E_pen: {E_pen:.4f}, E_smooth: {E_smooth:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
                )

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            scheduler.step()

            current_loss = cost.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_opt_qpos = opt_qpos.detach().clone()
                best_max_pene = max_pene.detach().clone()

            with torch.no_grad():
                if helper.ignore_joint_indices.numel() > 0:
                    opt_qpos[..., helper.ignore_joint_indices] = self.robot_init_qpos[helper.ignore_joint_indices]

        with torch.no_grad():
            if best_loss < cost.item():
                opt_qpos.data.copy_(best_opt_qpos.data)

        full_traj_qpos = torch.cat([start_qpos, opt_qpos, end_qpos], dim=-2)
        return full_traj_qpos.detach(), best_max_pene.detach()

    @torch.no_grad()
    def get_lifting_target_qpos(
        self,
        post_grasp_qpos: Float[torch.Tensor, "n_scenes n_grasps num_dofs"],
    ) -> Float[torch.Tensor, "n_scenes n_grasps num_dofs"]:
        if self.robot_config.base_type == "floating":
            target_qpos = post_grasp_qpos.clone()
            target_qpos[..., 2] += self.cfg.lifting_dist
            return target_qpos.detach()
        else:
            # Lock start_from_grasp_dof_indices at post-grasp values so the lift can't be undone by that freeze.
            grasp_link_poses = self.robot_helper.forward_kinematics_torch(post_grasp_qpos)
            grasp_hand_link_poses = grasp_link_poses[
                ..., self.robot_helper.link_names.index(self.robot_config.hand_base_link_name), :, :
            ]
            target_hand_link_poses = grasp_hand_link_poses.clone()
            target_hand_link_poses[..., 2, 3] += self.cfg.lifting_dist
            # IK targets are relative to arm_base_link (root link if None), not the robot root (e.g. G1's torso).
            arm_base_pose = (
                self.robot_helper.forward_kinematics_torch(self.robot_helper.init_qpos[None, :])[
                    0, self.robot_helper.link_names.index(self.robot_config.arm_base_link_name)
                ]
                if self.robot_config.arm_base_link_name is not None
                else self.robot_helper.root_pose_mat
            )
            target_hand_link_poses_in_armbase = inverse_tf_mat(arm_base_pose) @ target_hand_link_poses
            original_shape = target_hand_link_poses_in_armbase.shape[:-2]
            num_targets = int(target_hand_link_poses_in_armbase.numel() // 16)
            target_poses_for_ik = target_hand_link_poses_in_armbase.reshape(num_targets, 4, 4)
            init_arm_qpos = post_grasp_qpos[..., self.robot_helper.arm_joint_indices].reshape(num_targets, -1)
            target_arm_qpos, _ = self.robot_helper.inverse_kinematics(
                target_poses_for_ik,
                init_qpos=init_arm_qpos,
                init_sample_range=0.1,
                locked_joint_indices=self.cfg.start_from_grasp_dof_indices,
            )
            target_arm_qpos = target_arm_qpos.reshape(*original_shape, -1)
            target_qpos = post_grasp_qpos.clone()
            target_qpos[..., self.robot_helper.arm_joint_indices] = target_arm_qpos
            return target_qpos.detach()
