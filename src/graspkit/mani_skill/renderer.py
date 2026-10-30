import gc
import logging
from collections import defaultdict
from typing import List, Optional

import gymnasium as gym
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from rich.progress import track
from robokit.xform.utils import coord_conversion
from robokit.xform.warp.torch_wrappers import inverse_tf_mat, pose7_to_tf_mat, tf_mat_to_pose7, transform_points

from graspkit.configs.experiment_config import RendererConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.mani_skill.agents.config_agent import register_config_agent
from graspkit.mani_skill.utils.reindex_utils import reindex_qpos_ours_to_maniskill
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.point_cloud_utils import (
    crop_and_resample_point_cloud,
    extract_point_cloud_segments_inplace,
    remove_point_cloud_segments_inplace,
)
from graspkit.utils.tensor_utils import batchify_tensor, padding_tensor
from graspkit.utils.tui_utils import log_once

logger = logging.getLogger("graspkit")

_DEFAULT_CAMERA_POSITION = (0.56, 0.07, 0.37)
_DEFAULT_CAMERA_QUATERNION = (0.02, 0.21, 0.0, -0.98)


class MsRenderer:
    """ManiSkill `DexGrasp` training-data renderer that never steps physics, built by `graspkit.envs.build_renderer`."""

    def __init__(
        self,
        config: RendererConfig,
        robot_config: RobotConfig,
        device: torch.device,
    ):
        self.cfg = config

        ms = self.cfg.env_config.ms_env_config
        uid = register_config_agent(robot_config, self.cfg.env_config.cameras)
        gpu_backend = f"gpu:{device.index}" if device.type == "cuda" and device.index is not None else "gpu"
        self.env = gym.make(
            "DexGrasp",
            config=self.cfg.env_config,
            robot_uids=uid if ms.robot_uid is None else ms.robot_uid,
            num_envs=self.cfg.num_envs,
            sim_backend=gpu_backend,
            render_backend=gpu_backend,
            obs_mode=self.cfg.obs_mode,
            control_mode="pd_joint_pos",
        )
        self.device = device
        self.robot_config = robot_config
        self.robot_helper = GraspRobot.from_config(self.robot_config, device=device)
        # The ManiSkill agent has ALL joints while pipeline qpos is chain-space: scatter chain qpos into the full robot.
        self._full_helper = GraspRobot.from_config(self.robot_config, device=device, prune=False)
        self._chain_to_full = torch.tensor(
            [self._full_helper.actuated_joint_names.index(n) for n in self.robot_helper.actuated_joint_names],
            device=device,
        )
        self.robot_root_pose = torch.tensor(
            (*self.robot_config.init_position, *self.robot_config.init_quaternion), device=device
        )
        zero_qpos = self.robot_helper.spec.get_tensors_torch(device=device).zero_q
        self.default_robot_qpos = zero_qpos.to(device)
        self.default_camera_pose = torch.tensor(
            (*_DEFAULT_CAMERA_POSITION, *_DEFAULT_CAMERA_QUATERNION),
            device=device,
        )
        self.batch = 0

    @torch.no_grad()
    def render(
        self,
        scene_descriptions: List[SceneDescription],
        camera_poses: Optional[Float[torch.Tensor, "num_scenes num_views 7"]] = None,
        robot_qpos: Optional[Float[torch.Tensor, "*num_scenes num_steps num_joints"]] = None,
        robot_root_poses: Optional[Float[torch.Tensor, "*num_scenes 7"]] = None,
        object_poses: Optional[Float[torch.Tensor, "*num_scenes num_steps 7"]] = None,
    ):
        num_scenes = len(scene_descriptions)
        native_cameras = bool(self.cfg.env_config.cameras)
        assert not native_cameras or camera_poses is None
        if camera_poses is None:
            camera_poses = self.default_camera_pose.expand((num_scenes, 1, 7))
        num_views = camera_poses.shape[1]
        if robot_root_poses is None:
            robot_root_poses = self.robot_root_pose
        if robot_qpos is None:
            robot_qpos = self.default_robot_qpos.unsqueeze(0)
        if (
            len(scene_descriptions) > 1
            and num_views > 1
            and not (self.cfg.num_envs % num_views == 0 or num_views % self.cfg.num_envs == 0)
        ):
            log_once(
                f"For {len(scene_descriptions)} scenes with {num_views} views, "
                f"using {self.cfg.num_envs} environments will require many reconfigurations. "
                f"For optimal performance, set num_envs to a value where num_views % num_envs == 0 or num_envs % num_views == 0.",
                level=logging.WARNING,
                logger=logger,
            )
        flatten_scene_desc_indices = [i for i in range(len(scene_descriptions)) for _ in range(num_views)]
        flatten_camera_poses = rearrange(camera_poses, "num_scenes num_views d -> (num_scenes num_views) d")
        flatten_root_poses = repeat(
            robot_root_poses.expand((num_scenes, 7)),
            "num_scenes d -> (num_scenes num_views) d",
            num_views=num_views,
        )
        flatten_robot_qpos = repeat(
            robot_qpos.expand((num_scenes, -1, -1)),
            "num_scenes num_steps d -> (num_scenes num_views) num_steps d",
            num_views=num_views,
        )
        if object_poses is not None:
            flatten_object_poses = repeat(
                object_poses.expand((num_scenes, -1, 7)),
                "num_scenes num_steps d -> (num_scenes num_views) num_steps d",
                num_views=num_views,
            )
        else:
            flatten_object_poses = [None] * len(flatten_scene_desc_indices)

        all_obs = {
            "sensor_data": defaultdict(lambda: defaultdict(list)),
            "sensor_param": defaultdict(lambda: defaultdict(list)),
            "pointcloud": defaultdict(list),
        }
        prev_batch_scene_desc_indices = None

        self.batch += 1
        if self.batch % 50 == 0:
            gc.collect()
        for (
            batch_scene_desc_indices,
            batch_camera_poses,
            batch_root_pose,
            batch_robot_qpos,
            batch_object_poses,
        ) in track(
            zip(
                batchify_tensor(flatten_scene_desc_indices, self.cfg.num_envs),
                batchify_tensor(flatten_camera_poses, self.cfg.num_envs),
                batchify_tensor(flatten_root_poses, self.cfg.num_envs),
                batchify_tensor(flatten_robot_qpos, self.cfg.num_envs),
                batchify_tensor(flatten_object_poses, self.cfg.num_envs),
            ),
            description="Rendering",
            total=len(flatten_scene_desc_indices),
            disable=True,
        ):
            batch_scene_desc_indices, _ = padding_tensor(batch_scene_desc_indices, self.cfg.num_envs)
            batch_camera_poses, batch_mask = padding_tensor(batch_camera_poses, self.cfg.num_envs)
            batch_root_pose, _ = padding_tensor(batch_root_pose, self.cfg.num_envs)
            batch_robot_qpos, _ = padding_tensor(batch_robot_qpos, self.cfg.num_envs)
            batch_object_poses, _ = padding_tensor(batch_object_poses, self.cfg.num_envs)
            num_free = batch_robot_qpos.shape[-1] - self.robot_helper.num_actuated_joints  # 6 if floating
            full_q = self._full_helper.init_qpos.to(batch_robot_qpos)
            full_q = full_q.expand(*batch_robot_qpos.shape[:-1], -1).clone()
            full_q[..., self._chain_to_full.to(full_q.device)] = batch_robot_qpos[..., num_free:]
            full_q = torch.cat([batch_robot_qpos[..., :num_free], full_q], dim=-1)
            batch_robot_qpos_ms = reindex_qpos_ours_to_maniskill(full_q, self._full_helper, self.env)

            reconfigure = (
                prev_batch_scene_desc_indices is None or prev_batch_scene_desc_indices != batch_scene_desc_indices
            )
            prev_batch_scene_desc_indices = batch_scene_desc_indices

            batch_traj_obs = {
                "sensor_data": defaultdict(lambda: defaultdict(list)),
                "sensor_param": defaultdict(lambda: defaultdict(list)),
                "pointcloud": defaultdict(list),
                "camera_poses": [],  # track per-step camera poses actually used
            }
            if reconfigure:
                batch_frame_obs, _ = self.env.reset(
                    options={
                        "scene_descriptions": [scene_descriptions[i] for i in batch_scene_desc_indices],
                        "camera_poses": batch_camera_poses,
                        "root_poses": batch_root_pose,
                        "qpos": batch_robot_qpos_ms[..., 0, :],
                        "reconfigure": True,
                    },
                    seed=self.cfg.seed,
                )
            else:
                batch_frame_obs, _ = self.env.reset(
                    options={
                        "camera_poses": batch_camera_poses,
                        "root_poses": batch_root_pose,
                        "qpos": batch_robot_qpos_ms[..., 0, :],
                    },
                    seed=self.cfg.seed,
                )

            for step_idx in range(batch_robot_qpos.shape[1]):
                if step_idx > 0:  # NOTE directly use batch_frame_obs for the 0-th step
                    self.env.unwrapped.agent.robot.set_qpos(batch_robot_qpos_ms[..., step_idx, :])
                    self.env.unwrapped.target_actor.set_pose(batch_object_poses[..., step_idx, :])
                    self.env.unwrapped.scene._gpu_apply_all()
                    self.env.unwrapped.scene.px.gpu_update_articulation_kinematics()
                    self.env.unwrapped.scene._gpu_fetch_all()
                    batch_frame_obs = self.env.unwrapped.get_obs()

                if self.cfg.obs_mode == "pointcloud":
                    batch_frame_obs["sensor_data"] = self.env.unwrapped._get_obs_sensor_data()
                frame_mask = batch_mask
                frame_scene_indices = batch_scene_desc_indices
                frame_camera_poses = batch_camera_poses
                frame_num_views = 1
                if native_cameras:
                    sensor_uids = list(self.env.unwrapped._sensors)
                    frame_num_views = len(sensor_uids)
                    frame_mask = batch_mask.repeat_interleave(frame_num_views)
                    frame_scene_indices = [i for i in batch_scene_desc_indices for _ in sensor_uids]
                    for data_type in ("sensor_data", "sensor_param"):
                        sensors = batch_frame_obs[data_type]
                        batch_frame_obs[data_type] = {
                            "base_camera": {
                                key: torch.stack([sensors[uid][key] for uid in sensor_uids], dim=1).flatten(0, 1)
                                for key in sensors[sensor_uids[0]]
                            }
                        }
                    sapien_to_gl = torch.eye(4, device=self.device)
                    sapien_to_gl[:3, :3] = coord_conversion("sapien", "opengl", return_tensors="pt").to(self.device)
                    frame_camera_poses = tf_mat_to_pose7(
                        batch_frame_obs["sensor_param"]["base_camera"]["cam2world_gl"].to(self.device) @ sapien_to_gl
                    )
                    if "pointcloud" in batch_frame_obs:
                        batch_frame_obs["pointcloud"] = {
                            key: values.reshape(self.cfg.num_envs, frame_num_views, -1, values.shape[-1]).flatten(0, 1)
                            for key, values in batch_frame_obs["pointcloud"].items()
                        }
                for data_type in ["sensor_data", "sensor_param"]:
                    for sensor_uid, sensor_data in batch_frame_obs[data_type].items():
                        for k, v in sensor_data.items():
                            if k == "rgb" and isinstance(v, list):
                                v = torch.stack(v, dim=0)
                            batch_traj_obs[data_type][sensor_uid][k].append(v[frame_mask].cpu())

                if "pointcloud" in batch_frame_obs:
                    segments = self.cfg.extract_segments or self.cfg.remove_segments
                    ids = []
                    if "table" in segments:
                        ids.append(
                            self.env.unwrapped.table_segmentation_id.repeat_interleave(frame_num_views).unsqueeze(-1)
                        )
                    if "object" in segments:
                        ids.append(
                            self.env.unwrapped.target_actor_segmentation_id.repeat_interleave(
                                frame_num_views
                            ).unsqueeze(-1)
                        )
                    if "robot" in segments:
                        ids.append(self.env.unwrapped.robot_segmentation_id.unsqueeze(0).repeat(len(frame_mask), 1))
                    if ids:
                        all_ids = torch.cat(ids, dim=1)
                        if self.cfg.extract_segments:
                            extract_point_cloud_segments_inplace(batch_frame_obs["pointcloud"], all_ids, use_cpu=False)
                        else:
                            remove_point_cloud_segments_inplace(batch_frame_obs["pointcloud"], all_ids, use_cpu=False)

                    if self.cfg.point_cloud_frame == "camera":
                        crop_bounds = self.cfg.points_crop_bounds_camera
                        current_camera_poses = frame_camera_poses[frame_mask]

                        camera_poses_inv = inverse_tf_mat(pose7_to_tf_mat(current_camera_poses))
                        sapien_to_opencv_rot = coord_conversion("sapien", "opencv", return_tensors="pt")
                        sapien_to_opencv_tf = torch.eye(4, device=camera_poses_inv.device)
                        sapien_to_opencv_tf[:3, :3] = sapien_to_opencv_rot
                        camera_poses_opencv = torch.matmul(sapien_to_opencv_tf, camera_poses_inv)

                        active_xyzw = batch_frame_obs["pointcloud"]["xyzw"][frame_mask]
                        transformed_xyz = transform_points(active_xyzw[..., :3], camera_poses_opencv)
                        batch_frame_obs["pointcloud"]["xyzw"][frame_mask] = torch.cat(
                            [transformed_xyz, active_xyzw[..., 3:]], dim=-1
                        )
                        # record the used camera poses (in world frame 7D)
                        batch_traj_obs["camera_poses"].append(current_camera_poses.cpu())
                    # If not using camera frame, do not transform world-frame point clouds here
                    else:
                        # World-frame crop bounds are relative to the scene's table, else off-origin scenes crop to 0.
                        base_bounds = torch.tensor(
                            self.cfg.points_crop_bounds, device=self.device, dtype=torch.float32
                        )  # (3, 2)
                        if self.cfg.env_config.floating_object:
                            crop_bounds = base_bounds.unsqueeze(0).expand(len(frame_scene_indices), -1, -1)
                        else:
                            table_positions = torch.tensor(
                                [scene_descriptions[i].boxes["table"].position for i in frame_scene_indices],
                                device=self.device,
                                dtype=torch.float32,
                            )  # (num_envs, 3)
                            crop_bounds = base_bounds.unsqueeze(0) + table_positions.unsqueeze(-1)  # (num_envs, 3, 2)
                        # Even without camera-frame rendering, record the camera pose per step (no perturbation applied)
                        batch_traj_obs["camera_poses"].append(frame_camera_poses[frame_mask].cpu())

                    batch_pcd_obs = crop_and_resample_point_cloud(
                        batch_frame_obs["pointcloud"], crop_bounds, self.cfg.num_points
                    )

                    batch_traj_obs["pointcloud"]["xyz"].append(batch_pcd_obs["xyz"][frame_mask].cpu())
                    batch_traj_obs["pointcloud"]["rgb"].append(batch_pcd_obs["rgb"][frame_mask].cpu())
                    batch_traj_obs["pointcloud"]["seg"].append(batch_pcd_obs["segmentation"][frame_mask].cpu())
                    batch_traj_obs["pointcloud"]["seg_id"].append(
                        self.env.unwrapped.target_actor_segmentation_id.repeat_interleave(frame_num_views)[frame_mask]
                        .unsqueeze(1)
                        .cpu()
                    )

            for data_type in ["sensor_data", "sensor_param"]:
                for sensor_uid, sensor_data in batch_traj_obs[data_type].items():
                    for k, v in sensor_data.items():
                        batch_traj_obs[data_type][sensor_uid][k] = torch.stack(v, dim=1)
            for k, v in batch_traj_obs["pointcloud"].items():
                batch_traj_obs["pointcloud"][k] = torch.stack(v, dim=1).cpu()
            if batch_traj_obs["camera_poses"]:
                batch_traj_obs["camera_poses"] = torch.stack(batch_traj_obs["camera_poses"], dim=1)

            for data_type in ["sensor_data", "sensor_param"]:
                for sensor_uid, sensor_data in batch_traj_obs[data_type].items():
                    for k, v in sensor_data.items():
                        all_obs[data_type][sensor_uid][k].append(v)
            for k, v in batch_traj_obs["pointcloud"].items():
                all_obs["pointcloud"][k].append(v)
            if isinstance(batch_traj_obs["camera_poses"], torch.Tensor):
                all_obs.setdefault("camera_poses", []).append(batch_traj_obs["camera_poses"].cpu())

        for data_type in ["sensor_data", "sensor_param"]:
            for sensor_uid, sensor_data in all_obs[data_type].items():
                for k, v in sensor_data.items():
                    all_obs[data_type][sensor_uid][k] = torch.cat(v, dim=0)
        for k, v in all_obs["pointcloud"].items():
            all_obs["pointcloud"][k] = torch.cat(v, dim=0)
        if "camera_poses" in all_obs:
            all_obs["camera_poses"] = torch.cat(all_obs["camera_poses"], dim=0)

        return all_obs

    def close(self):
        self.env.close()
