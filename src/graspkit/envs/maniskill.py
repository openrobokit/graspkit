import gc
import json
import logging
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from einops import rearrange
from rich.progress import track
from robokit.xform.warp.torch_wrappers import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from torch.types import Device

from graspkit.configs.experiment_config import EnvConfig, RendererConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs.base import BaseEnv, Rollout
from graspkit.mani_skill.agents.config_agent import register_config_agent
from graspkit.mani_skill.renderer import MsRenderer
from graspkit.mani_skill.utils.reindex_utils import (
    reindex_qpos_maniskill_to_ours,
    reindex_qpos_ours_to_maniskill,
)
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.point_cloud_utils import crop_and_resample_point_cloud
from graspkit.utils.tensor_utils import batchify_tensor, padding_tensor
from graspkit.utils.tui_utils import log_once
from graspkit.utils.xform_utils import compute_camera_world_pose_7d

logger = logging.getLogger("graspkit")


class MsEnv(BaseEnv):
    """ManiSkill-backed grasp env with two usage modes:

    - SR eval: `evaluate` runs the batched rollout with the default `state_dict` obs (no render)
    - deploy: `reset`/`get_qpos`/`step_qpos`/`get_pointcloud` drive a single-instance loop; construct with
      `obs_mode="pointcloud"`, the submitted RGB mask selects the organized point cloud
    """

    def __init__(
        self,
        cfg: EnvConfig,
        robot_config: RobotConfig,
        device: Device,
        obs_mode: str = "state_dict",
        pc_sample_size: int = 4096,
        renderer_cfg: Optional[RendererConfig] = None,
        enable_rendering: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.robot_config = robot_config
        self.device = device
        self.pc_sample_size = pc_sample_size
        self.renderer_cfg = renderer_cfg  # when set, get_pointcloud replicates Renderer.render exactly
        self.renderer = self.camera_poses = None
        if renderer_cfg is not None and obs_mode == "state_dict":
            # Without sensors of its own, `observe` re-renders the state in a separate Renderer env, as training did.
            rcfg = replace(renderer_cfg, num_envs=1)
            if rcfg.camera_poses_path is not None:  # calibrated views: one env per view, real intrinsics
                poses = torch.tensor(
                    json.loads(Path(rcfg.camera_poses_path).read_text())["T_base_camera"], device=device
                )
                self.camera_poses = compute_camera_world_pose_7d(
                    poses.new_tensor((0, 0, 0, 1, 0, 0, 0)), poses.reshape(-1, 4, 4)
                ).unsqueeze(0)
                k = json.loads(Path(rcfg.camera_intrinsic_path).read_text())["intrinsic"]
                s = rcfg.render_downscale
                rcfg = replace(
                    rcfg,
                    num_envs=self.camera_poses.shape[1],
                    env_config=replace(
                        rcfg.env_config,
                        camera_width=int(k[5] / s),
                        camera_height=int(k[4] / s),
                        camera_intrinsic=((k[0] / s, 0, k[2] / s), (0, k[1] / s, k[3] / s), (0, 0, 1)),
                    ),
                )
            self.renderer = MsRenderer(rcfg, robot_config, device)
        self._scenes = []
        self._last_obs = None
        self._infos = None
        uid = register_config_agent(robot_config, cfg.cameras)
        gpu_backend = f"gpu:{device.index}" if device.type == "cuda" and device.index is not None else "gpu"
        self._ms_env = gym.make(
            "DexGrasp",
            config=cfg,
            robot_uids=uid if cfg.ms_env_config.robot_uid is None else cfg.ms_env_config.robot_uid,
            num_envs=self.cfg.num_envs,
            sim_backend=gpu_backend,
            render_backend=gpu_backend if enable_rendering else "none",
            obs_mode=obs_mode,
            control_mode="pd_joint_pos",
        )
        self.robot_helper = GraspRobot.from_config(self.robot_config, device=device)
        # ManiSkill agent has ALL joints, pipeline qpos is chain-space: scatter in, gather out.
        self._full_helper = GraspRobot.from_config(self.robot_config, device=device, prune=False)
        self._chain_to_full = torch.tensor(
            [self._full_helper.actuated_joint_names.index(n) for n in self.robot_helper.actuated_joint_names],
            device=device,
        )
        # Static base: floating pipeline robot on a hand-only agent; base is per-env root poses, not PD-driven joints.
        num_ms_joints = len(self._ms_env.unwrapped.agent.robot.active_joints)
        self._static_base = (
            robot_config.base_type == "floating" and num_ms_joints == self._full_helper.num_actuated_joints
        )

    def _base_to_root_pose(self, base: torch.Tensor) -> torch.Tensor:
        """Floating-base qpos[:6] ([xyz, euler_rxyz]) -> (..., 7) [xyz, quat_wxyz] root pose."""
        quat = matrix_to_quaternion(euler_angles_to_matrix(base[..., 3:6], "rxyz"))
        return torch.cat([base[..., :3], quat], dim=-1)

    def _qpos_ours_to_ms(self, q: torch.Tensor) -> torch.Tensor:
        num_free = q.shape[-1] - self.robot_helper.num_actuated_joints  # 6 for floating-base qpos
        if self._static_base:
            q = q[..., num_free:]  # base handled via root_poses at reset
            num_free = 0
        full_q = self._full_helper.init_qpos.to(q).expand(*q.shape[:-1], -1).clone()
        full_q[..., self._chain_to_full.to(q.device)] = q[..., num_free:]
        full_q = torch.cat([q[..., :num_free], full_q], dim=-1)
        return reindex_qpos_ours_to_maniskill(full_q, self._full_helper, self._ms_env)

    def _qpos_ms_to_ours(self, q_ms: torch.Tensor) -> torch.Tensor:
        full_q = reindex_qpos_maniskill_to_ours(q_ms, self._full_helper, self._ms_env)
        num_free = full_q.shape[-1] - self._full_helper.num_actuated_joints  # 6 for floating-base qpos
        chain_idx = self._chain_to_full.to(full_q.device) + num_free
        q = torch.cat([full_q[..., :num_free], full_q[..., chain_idx]], dim=-1)
        if self._static_base:  # rebuild the 6 base dims from the agent's (static) root pose
            pose = self._ms_env.unwrapped.agent.robot.pose.raw_pose.to(q)  # (B, 7) [xyz, quat_wxyz]
            euler = matrix_to_euler_angles(quaternion_to_matrix(pose[..., 3:]), "rxyz")
            q = torch.cat([pose[..., :3], euler, q], dim=-1)
        return q

    # ------------------------------------------------------------------ deploy primitives
    def reset(
        self,
        scene: SceneDescription | list[SceneDescription] | None = None,
        init_qpos: Optional[torch.Tensor] = None,
        root_pose: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
    ):
        """One env per scene in `scene` (a scene or a list, up to `num_envs`); `None` keeps the loaded scenes."""
        options = {}
        if scene is not None:
            self._scenes = scene if isinstance(scene, list) else [scene]
            options["scene_descriptions"] = self._scenes
            options["reconfigure"] = True
        if init_qpos is not None:
            q = init_qpos if init_qpos.dim() == 2 else init_qpos.unsqueeze(0)
            options["qpos"] = self._qpos_ours_to_ms(q).contiguous()
        if root_pose is not None:
            rp = root_pose if root_pose.dim() == 2 else root_pose.unsqueeze(0)
            options["root_poses"] = rp.contiguous()
        if camera_pose is not None:
            # Calibrated world-frame camera pose (== render_data.py); else falls back to the renderer's default mount.
            cp = camera_pose if camera_pose.dim() == 2 else camera_pose.unsqueeze(0)
            options["camera_poses"] = cp.contiguous()
        self._last_obs, _ = self._ms_env.reset(options=options)

    def get_qpos(self) -> torch.Tensor:
        qpos_ms = self._ms_env.unwrapped.agent.robot.get_qpos()
        return self._qpos_ms_to_ours(qpos_ms)

    def step_qpos(self, q: torch.Tensor):
        q = q if q.dim() == 2 else q.unsqueeze(0)
        q_ms = self._qpos_ours_to_ms(q)
        ms_env = self._ms_env.unwrapped
        ms_env._step_action(q_ms)
        ms_env._elapsed_steps += 1

    def get_info(self) -> dict[str, torch.Tensor]:
        return self._ms_env.unwrapped.get_info()

    def get_object_pose(self) -> torch.Tensor:
        return self._ms_env.unwrapped.target_actor.pose.raw_pose

    def set_object_pose(self, pose: torch.Tensor):
        scene = self._ms_env.unwrapped.scene
        self._ms_env.unwrapped.target_actor.set_pose(pose)
        scene._gpu_apply_all()
        scene._gpu_fetch_all()

    def observe(self) -> dict[str, torch.Tensor]:
        scenes = [
            replace(
                scene,
                meshes={
                    **scene.meshes,
                    scene.target_actor_id: replace(
                        scene.meshes[scene.target_actor_id], position=tuple(p[:3]), quaternion=tuple(p[3:]), bounds=None
                    ),
                },
            )
            for scene, p in zip(self._scenes, self.get_object_pose().cpu().tolist(), strict=True)
        ]
        b = len(scenes)
        rendered = self.renderer.render(
            scenes,
            camera_poses=None if self.camera_poses is None else self.camera_poses.expand(b, -1, -1),
            robot_root_poses=self._ms_env.unwrapped.agent.robot.pose.raw_pose,
            robot_qpos=self.get_qpos().unsqueeze(1),
        )
        rgb = rendered["sensor_data"]["base_camera"]["rgb"][:, 0]
        return {
            "rgb": rgb.reshape(b, -1, *rgb.shape[1:]),
            "xyz": rendered["pointcloud"]["xyz"][:, 0].reshape(b, -1, 3),
            "seg": rendered["pointcloud"]["seg"][:, 0].reshape(b, -1, 1),
            "seg_id": rendered["pointcloud"]["seg_id"][:, 0, 0].reshape(b, -1)[:, 0],
            "camera_poses": rendered["camera_poses"].reshape(b, -1, 7),
        }

    def get_rgb(self) -> np.ndarray:
        h, w = self.cfg.camera_height, self.cfg.camera_width
        rgb = self._last_obs["pointcloud"]["rgb"][0]
        assert rgb.shape[0] == h * w, "deploy supports exactly one camera"
        return rgb.reshape(h, w, 3).cpu().numpy()

    def get_gt_mask(self) -> np.ndarray:
        h, w = self.cfg.camera_height, self.cfg.camera_width
        seg = self._last_obs["pointcloud"]["segmentation"][0, :, 0]
        target_id = self._ms_env.unwrapped.target_actor_segmentation_id[0]
        return (seg == target_id).reshape(h, w).cpu().numpy()

    def get_pointcloud(self, mask: np.ndarray) -> torch.Tensor:
        h, w = self.cfg.camera_height, self.cfg.camera_width
        assert mask.shape == (h, w)
        pc = dict(self._last_obs["pointcloud"])  # {xyzw, rgb, segmentation}, world frame
        pc["xyzw"] = pc["xyzw"].clone()
        keep = torch.from_numpy(mask.reshape(-1)).to(self.device).bool()
        pc["xyzw"][0, ~keep, 3] = 0
        if self.renderer_cfg is not None:
            assert self.renderer_cfg.point_cloud_frame == "world", "only world-frame render is aligned here"
            base = torch.tensor(self.renderer_cfg.points_crop_bounds, device=self.device, dtype=torch.float32)  # (3,2)
            if self.cfg.floating_object:
                bounds = base
            else:
                table_pos = torch.tensor(
                    self._scenes[0].boxes["table"].position, device=self.device, dtype=torch.float32
                )
                bounds = base + table_pos.unsqueeze(-1)  # table-relative, like the renderer
            return crop_and_resample_point_cloud(pc, bounds, self.renderer_cfg.num_points)[
                "xyz"
            ]  # (B, num_points, 3), world frame
        xyzw = pc["xyzw"]
        pts = xyzw[0, xyzw[0, :, 3] > 0, :3]
        if pts.shape[0] > self.pc_sample_size:
            pts = pts[torch.randperm(pts.shape[0], device=pts.device)[: self.pc_sample_size]]
        return pts.unsqueeze(0)

    # ------------------------------------------------------------------ batched SR rollout
    @torch.no_grad()
    def evaluate(self, batch: GraspKitData, record: bool = False) -> dict[str, torch.Tensor]:
        """Rollout full_traj_qpos, return per-metric infos (B,K); if record, save seed-0 rollouts to last_rollouts."""
        self.last_rollouts = []
        scene_descriptions = batch.scene_descriptions
        full_traj_qpos = batch.full_traj_qpos
        num_samples = full_traj_qpos.shape[1]
        if (
            len(scene_descriptions) * num_samples > self.cfg.num_envs
            and num_samples > 1
            and not (self.cfg.num_envs % num_samples == 0 or num_samples % self.cfg.num_envs == 0)
        ):
            log_once(
                f"For {len(scene_descriptions)} scenes with {num_samples} samples, "
                f"using {self.cfg.num_envs} environments will require many reconfigurations. "
                f"For optimal performance, set num_envs to a value where num_samples % num_envs == 0 or num_envs % num_samples == 0.",
                level=logging.WARNING,
                logger=logger,
            )
        flatten_scene_desc_indices = [i for i in range(len(scene_descriptions)) for _ in range(num_samples)]
        if len(flatten_scene_desc_indices) <= self.cfg.num_envs // 2:
            log_once(
                f"Evaluating {len(flatten_scene_desc_indices)} scenes with {self.cfg.num_envs} environments is "
                f"likely to be inefficient. Consider reducing num_envs or increasing num_samples.",
                level=logging.WARNING,
                logger=logger,
            )
        flatten_qpos = rearrange(
            full_traj_qpos,
            "num_scenes num_samples num_steps num_dofs -> (num_scenes num_samples) num_steps num_dofs",
        )
        if self._static_base:
            # Pin the base at the FINAL (grasp) pose; grasp-only test, no approach sweep or base PD shaking.
            flatten_root_poses = self._base_to_root_pose(flatten_qpos[:, -1, :6])
        else:
            flatten_root_poses = rearrange(
                self.robot_helper.root_pose_7d.expand((*full_traj_qpos.shape[:-2], 7)),
                "num_scenes num_samples num_dofs -> (num_scenes num_samples) num_dofs",
            ).to(full_traj_qpos.device)

        final_infos = None
        cur_seed_mask = torch.ones(len(flatten_scene_desc_indices), dtype=torch.bool, device=full_traj_qpos.device)

        for seed in range(self.cfg.num_seeds):
            infos_cur_seed = defaultdict(list)
            prev_batch_scene_desc_indices = None
            cur_seed_indices = torch.where(cur_seed_mask)[0]
            if len(cur_seed_indices) == 0:
                break
            for batch_indices in track(
                batchify_tensor(cur_seed_indices, self.cfg.num_envs),
                description="Evaluating",
                total=len(cur_seed_indices),
                disable=True,
            ):
                batch_indices, batch_mask = padding_tensor(batch_indices, self.cfg.num_envs)
                # Transfer once: indexing a Python list with each CUDA scalar synchronizes every environment.
                batch_scene_desc_indices = [flatten_scene_desc_indices[i] for i in batch_indices.cpu().tolist()]
                batch_qpos = flatten_qpos[batch_indices]
                batch_root_pose = flatten_root_poses[batch_indices]

                reconfigure = batch_scene_desc_indices != prev_batch_scene_desc_indices
                prev_batch_scene_desc_indices = batch_scene_desc_indices
                batch_qpos_ms = self._qpos_ours_to_ms(batch_qpos)
                if reconfigure:
                    self._ms_env.reset(
                        options={
                            "scene_descriptions": [scene_descriptions[i] for i in batch_scene_desc_indices],
                            "qpos": batch_qpos_ms[..., 0, :].contiguous(),
                            "root_poses": batch_root_pose.contiguous(),
                            "reconfigure": True,
                        },
                        seed=seed,
                    )
                else:
                    gc.collect()  # force gc to collect which releases most GPU memory
                    self._ms_env.reset(
                        options={
                            "qpos": batch_qpos_ms[..., 0, :].contiguous(),
                            "root_poses": batch_root_pose.contiguous(),
                        },
                        seed=seed,
                    )
                if self.cfg.enable_perturbation:
                    angles = [i * 2 * math.pi / self.cfg.num_perturbations for i in range(self.cfg.num_perturbations)]
                    force_magnitude = self.cfg.perturbation_magnitude
                    perturb_forces = [
                        torch.tensor(
                            [force_magnitude * math.cos(angle), force_magnitude * math.sin(angle), 0.0],
                            device=batch_qpos_ms.device,
                        )
                        for angle in angles
                    ]
                    perturb_step_start_cfg, perturb_step_end_cfg = self.cfg.perturbation_step_range
                    num_steps = batch_qpos_ms.shape[1]
                    perturb_step_start = (
                        perturb_step_start_cfg if perturb_step_start_cfg >= 0 else num_steps + perturb_step_start_cfg
                    )
                    perturb_step_end = (
                        perturb_step_end_cfg if perturb_step_end_cfg >= 0 else num_steps + perturb_step_end_cfg
                    )
                    perturb_step_start = max(0, min(perturb_step_start, num_steps - 1))
                    perturb_step_end = max(0, min(perturb_step_end, num_steps - 1))
                    if perturb_step_start > perturb_step_end:
                        perturb_step_start, perturb_step_end = perturb_step_end, perturb_step_start
                    interval_raw = (perturb_step_end - perturb_step_start) // self.cfg.num_perturbations
                    perturb_interval = max(1, interval_raw)

                record_this = record and seed == 0  # seed 0 covers every grasp exactly once
                rec_qpos, rec_obj = [], []
                for step in range(batch_qpos_ms.shape[1]):
                    step_qpos_ms = batch_qpos_ms[:, step]
                    if (
                        self.cfg.enable_perturbation
                        and perturb_step_start <= step <= perturb_step_end
                        and (step - perturb_step_start) % perturb_interval < self.cfg.perturbation_duration
                    ):
                        perturb_index = (step - perturb_step_start) // perturb_interval
                        perturb_index = min(perturb_index, self.cfg.num_perturbations - 1)
                        force = perturb_forces[perturb_index]
                        self._ms_env.unwrapped.target_actor.apply_force(force)
                    _, _, _, _, infos = self._ms_env.step(step_qpos_ms)
                    if record_this:
                        q_ours = self._qpos_ms_to_ours(self._ms_env.unwrapped.agent.robot.get_qpos())
                        rec_qpos.append(q_ours.cpu())  # (num_envs, dof) FK variables
                        rec_obj.append(self._ms_env.unwrapped.target_actor.pose.raw_pose.cpu())  # (num_envs, 7)
                for k, v in infos.items():
                    infos_cur_seed[k].append(v[batch_mask.to(v.device)])

                if record_this:
                    q_traj = torch.stack(rec_qpos, dim=0)  # (T, num_envs, dof)
                    o_traj = torch.stack(rec_obj, dim=0)  # (T, num_envs, 7)
                    for e in torch.where(batch_mask)[0].tolist():  # all real (non-padded) envs
                        self.last_rollouts.append(
                            Rollout(
                                qpos=q_traj[:, e].numpy(),
                                obj_pose=o_traj[:, e].numpy(),
                                scene_id=scene_descriptions[batch_scene_desc_indices[e]].scene_id,
                                success=bool(infos["success"][e]),
                                grasp_index=int(batch_indices[e]) % num_samples,
                            )
                        )

            infos_cur_seed = {k: torch.cat(v, dim=0) for k, v in infos_cur_seed.items()}
            if final_infos is None:
                final_infos = infos_cur_seed
            else:
                for k, v in infos_cur_seed.items():
                    final_infos[k][cur_seed_mask] = v
            # keep only envs that succeeded this seed, and re-verify their robustness next round
            cur_seed_mask[cur_seed_indices] = torch.logical_and(
                cur_seed_mask[cur_seed_indices], infos_cur_seed["success"]
            ).clone()

        return {
            k: rearrange(v, "(num_scenes num_samples) -> num_scenes num_samples", num_scenes=len(scene_descriptions))
            for k, v in final_infos.items()
        }

    def close(self):
        self._ms_env.close()
        if self.renderer is not None:
            self.renderer.close()
