"""
Usage:
uv run scripts/stages/obs_gen/render_data.py g1_inspire_dex1b_tabletop_mini
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import json
import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from rich.progress import track
from robokit.xform.utils import coord_conversion

from graspkit.configs.experiment_config import ExperimentConfig, RendererConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.scene.scene_description import SceneDescription
from graspkit.envs import build_renderer
from graspkit.utils.seed_utils import seed_everything
from graspkit.utils.stage_utils import write_completion_marker
from graspkit.utils.tensor_utils import batchify_tensor
from graspkit.utils.xform_utils import compute_camera_world_pose_7d
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, pose7_to_tf_mat, tf_mat_to_pose7, rot_tl_to_tf_mat

_VISIBILITY_NEAR = 0.01
_VISIBILITY_FAR = 5.0

RENDER_COMPLETE_MARKER = ".render_complete"
PARKED_ROBOT_POSE_7D = (100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0)
logger = logging.getLogger("graspkit")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)


def _lookat_rot(
    eye: torch.Tensor,  # (..., 3) world
    target: torch.Tensor,  # (..., 3) world
    device: torch.device,
) -> torch.Tensor:
    """Sapien cam-to-world rotation (..., 3, 3) orienting the camera from eye to target."""
    forward = target - eye
    forward = forward / forward.norm(dim=-1, keepdim=True)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(forward)
    alt_up = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(forward)
    singular = (forward * world_up).sum(dim=-1, keepdim=True).abs() > 0.99
    up = torch.where(singular, alt_up, world_up)
    cam_x = torch.cross(forward, up, dim=-1)
    cam_x = cam_x / cam_x.norm(dim=-1, keepdim=True)
    cam_y = torch.cross(forward, cam_x, dim=-1)
    rot_opencv = torch.stack([cam_x, cam_y, forward], dim=-1)  # (..., 3, 3)
    sapien2opencv = coord_conversion("sapien", "opencv", return_tensors="pt").to(device)  # (3, 3)
    return rot_opencv @ sapien2opencv


def _build_camera_poses(
    scene_descriptions: list,
    base_camera_pose_7d: torch.Tensor,  # (V, 7) calibrated world poses
    renderer_cfg: RendererConfig,
    device: torch.device,
) -> torch.Tensor:
    """Per-scene camera poses (num_scenes, num_views, 7) sapien.

    - fixed: one view at the calibrated pose (or look-at the object).
    - sampled: camera_sample_num_views views; eye jitter +/-camera_sample_translation, look-at(object) or
      calibrated orientation, then +/-camera_sample_rotation rxyz jitter on top.
    """
    num_scenes = len(scene_descriptions)
    if renderer_cfg.camera_mode == "head_wrist":
        base_mat = pose7_to_tf_mat(base_camera_pose_7d.reshape(-1, 7)[0])
        tops = torch.tensor(
            [(*s.boxes["table"].position[:2], s.boxes["table"].position[2] + s.boxes["table"].size[2] / 2)
             for s in scene_descriptions], device=device, dtype=torch.float32,
        )  # fmt: skip
        head = torch.eye(4, device=device).repeat(num_scenes, 1, 1)
        rot_deg = torch.tensor(renderer_cfg.camera_sample_rotation, device=device, dtype=torch.float32)
        angles = (torch.rand(num_scenes, 3, device=device) * 2 - 1) * torch.deg2rad(rot_deg)
        head[:, :3, :3] = base_mat[:3, :3] @ euler_angles_to_matrix(angles, "rxyz")
        trans = torch.tensor(renderer_cfg.camera_sample_translation, device=device, dtype=torch.float32)
        head[:, :3, 3] = base_mat[:3, 3] + tops + (torch.rand(num_scenes, 3, device=device) * 2 - 1) * trans
        n = renderer_cfg.camera_sample_num_views
        target = torch.tensor(
            [s.meshes[s.target_actor_id].position for s in scene_descriptions], device=device, dtype=torch.float32
        )[:, None].expand(num_scenes, n, 3)
        head_dir = base_mat[:2, 3] / base_mat[:2, 3].norm()  # the head's side of the table, seen from above
        azim = torch.atan2(head_dir[1], head_dir[0]) + (
            torch.rand(num_scenes, n, device=device) * 2 - 1
        ) * math.radians(100)
        lo, hi = map(math.radians, renderer_cfg.wrist_elevation_deg)
        elev = lo + (hi - lo) * torch.rand(num_scenes, n, device=device)
        dist = renderer_cfg.wrist_distance[0] + (renderer_cfg.wrist_distance[1] - renderer_cfg.wrist_distance[0]) * torch.rand(num_scenes, n, device=device)  # fmt: skip
        eye = target + dist[..., None] * torch.stack(
            [elev.cos() * azim.cos(), elev.cos() * azim.sin(), elev.sin()], dim=-1
        )
        wrist = torch.eye(4, device=device).repeat(num_scenes, n, 1, 1)
        wrist[..., :3, :3] = _lookat_rot(eye, target, device)
        wrist[..., :3, 3] = eye
        return tf_mat_to_pose7(torch.cat([head[:, None], wrist], dim=1))
    sampled = renderer_cfg.camera_mode == "sampled"
    if not sampled and not renderer_cfg.look_at_object:
        return base_camera_pose_7d.reshape(1, -1, 7).expand(num_scenes, -1, -1)

    num_views = renderer_cfg.camera_sample_num_views if sampled else 1
    base_mat = pose7_to_tf_mat(base_camera_pose_7d.reshape(-1, 7)[0])  # (4, 4) sapien cam-to-world
    eye = base_mat[:3, 3].reshape(1, 1, 3).expand(num_scenes, num_views, 3).contiguous()
    if sampled:
        trans = torch.tensor(renderer_cfg.camera_sample_translation, device=device, dtype=torch.float32)
        eye = eye + (torch.rand_like(eye) * 2 - 1) * trans

    if renderer_cfg.look_at_object:
        object_centers = torch.tensor(
            [s.meshes[s.target_actor_id].position for s in scene_descriptions], device=device, dtype=torch.float32
        )  # (num_scenes, 3)
        target = object_centers.reshape(num_scenes, 1, 3).expand(num_scenes, num_views, 3)
        rot = _lookat_rot(eye, target, device)
    else:
        rot = base_mat[:3, :3].reshape(1, 1, 3, 3).expand(num_scenes, num_views, 3, 3).contiguous()

    if sampled and any(r != 0 for r in renderer_cfg.camera_sample_rotation):
        rot_deg = torch.tensor(renderer_cfg.camera_sample_rotation, device=device, dtype=torch.float32)
        angles = (torch.rand(num_scenes, num_views, 3, device=device) * 2 - 1) * torch.deg2rad(rot_deg)
        rot = rot @ euler_angles_to_matrix(angles, "rxyz")

    cam_to_world = torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(num_scenes, num_views, 4, 4).contiguous()
    cam_to_world[..., :3, :3] = rot
    cam_to_world[..., :3, 3] = eye
    return tf_mat_to_pose7(cam_to_world)


def _select_valid_view_indices(
    scene_descriptions: list,
    camera_poses_7d: torch.Tensor,  # (num_scenes, num_views, 7) sapien
    pcd_xyz: np.ndarray,  # (num_scenes, num_views, T, Npts, 3)
    intrinsic: torch.Tensor,  # (3, 3)
    image_size: Tuple[int, int],  # (width, height)
    renderer_cfg: RendererConfig,
) -> list:
    """Per-scene view indices passing the bbox visibility filter with a valid point (every view kept, no ranking)."""
    num_scenes, num_views = camera_poses_7d.shape[0], camera_poses_7d.shape[1]
    xyz_flat = pcd_xyz.reshape(num_scenes, num_views, -1, 3)
    valid_counts = (np.linalg.norm(xyz_flat, axis=-1) > 1e-6).sum(axis=-1)  # (num_scenes, num_views)
    width, height = image_size
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    sapien2opencv_rot = coord_conversion("sapien", "opencv", return_tensors="pt").to(camera_poses_7d.device)
    sapien2opencv = rot_tl_to_tf_mat(rot_mat=sapien2opencv_rot)
    base_bounds = renderer_cfg.points_crop_bounds
    chosen = []
    for scene_idx, scene_desc in enumerate(scene_descriptions):
        bounds = scene_desc.meshes[scene_desc.target_actor_id].get_bounds()
        x0, x1 = bounds[0]
        y0, y1 = bounds[1]
        z0, z1 = bounds[2]
        corners_world = torch.tensor(
            [
                [x0, y0, z0],
                [x0, y0, z1],
                [x0, y1, z0],
                [x0, y1, z1],
                [x1, y0, z0],
                [x1, y0, z1],
                [x1, y1, z0],
                [x1, y1, z1],
            ],
            dtype=torch.float32,
            device=camera_poses_7d.device,
        )
        if renderer_cfg.point_cloud_frame == "camera":
            crop_bounds = renderer_cfg.points_crop_bounds_camera
        elif renderer_cfg.env_config.floating_object:
            crop_bounds = base_bounds
        else:
            table_pos = scene_desc.boxes["table"].position
            crop_bounds = tuple((base_bounds[i][0] + table_pos[i], base_bounds[i][1] + table_pos[i]) for i in range(3))
        keep_views = np.zeros(num_views, dtype=np.bool_)
        for view_idx in range(num_views):
            cam_to_world_sapien = pose7_to_tf_mat(camera_poses_7d[scene_idx, view_idx])
            extrinsic_cv = sapien2opencv @ torch.linalg.inv(cam_to_world_sapien)
            ones = torch.ones((corners_world.shape[0], 1), device=corners_world.device)
            points_h = torch.cat([corners_world, ones], dim=-1)
            points_cam = (extrinsic_cv @ points_h.unsqueeze(-1)).squeeze(-1)[..., :3]
            z_cam = points_cam[:, 2]
            u = fx * points_cam[:, 0] / z_cam + cx
            v = fy * points_cam[:, 1] / z_cam + cy
            z_positive = torch.logical_and(z_cam > _VISIBILITY_NEAR, z_cam < _VISIBILITY_FAR)
            if not torch.any(z_positive):
                continue
            u_min, u_max = float(torch.min(u)), float(torch.max(u))
            v_min, v_max = float(torch.min(v)), float(torch.max(v))
            if u_max < 0 or v_max < 0 or u_min > (width - 1) or v_min > (height - 1):
                continue
            if renderer_cfg.point_cloud_frame == "camera":
                bbox_min = (
                    float(torch.min(points_cam[:, 0])),
                    float(torch.min(points_cam[:, 1])),
                    float(torch.min(points_cam[:, 2])),
                )
                bbox_max = (
                    float(torch.max(points_cam[:, 0])),
                    float(torch.max(points_cam[:, 1])),
                    float(torch.max(points_cam[:, 2])),
                )
            else:
                bbox_min = (x0, y0, z0)
                bbox_max = (x1, y1, z1)
            in_crop = all(not (bbox_max[i] < crop_bounds[i][0] or bbox_min[i] > crop_bounds[i][1]) for i in range(3))
            keep_views[view_idx] = in_crop
        valid = [int(v) for v in np.where(keep_views)[0] if valid_counts[scene_idx][v] > 0]
        chosen.append(valid)
    return chosen


def render_scene(exp_cfg: ExperimentConfig, robot_qpos_by_scene: dict[str, torch.Tensor] | None = None):
    renderer_cfg = exp_cfg.renderer
    assert robot_qpos_by_scene is None or renderer_cfg.camera_mode == "fixed"
    # Multi-view (sampled) produces multiple clouds per scene; per-view graspness is out of scope.
    assert not (exp_cfg.enable_contact_extraction and renderer_cfg.camera_mode == "sampled"), (
        "camera_mode='sampled' is not supported with enable_contact_extraction (per-view graspness)"
    )
    rank = exp_cfg.rank or 0
    world_size = exp_cfg.world_size or 1
    rendered_data_path = exp_cfg.rendered_data_path
    scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(exp_cfg.scene_description_path)

    scene_ids = sorted(list(scene_descriptions.keys()))
    scene_ids = scene_ids[rank::world_size]
    logger.info(f"[R{rank}/W{world_size}] Processing {len(scene_ids)} scenes into {rendered_data_path}")

    robot_config = exp_cfg.robot_config
    if renderer_cfg.camera_frame == "robot":
        rc = exp_cfg.traj_builder.fixed_base_robot_config  # arm/torso the camera is mounted on
        anchor_pose_7d = torch.tensor((*rc.init_position, *rc.init_quaternion), device=device, dtype=torch.float32)
    else:  # "world": calibrated pose is already world-frame
        anchor_pose_7d = torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), device=device, dtype=torch.float32)

    native_cameras = bool(renderer_cfg.env_config.cameras)
    if native_cameras:
        assert renderer_cfg.camera_mode == "fixed" and renderer_cfg.camera_poses_path is None
        camera_poses_7d = None
        camera_intrinsic = None
    else:
        with open(renderer_cfg.camera_poses_path, "r", encoding="utf-8") as f:
            camera_pose_data = json.load(f)
        if "T_base_camera" in camera_pose_data:
            T_base_camera = torch.as_tensor(camera_pose_data["T_base_camera"], device=device, dtype=torch.float32)
        else:
            T_base_camera = torch.linalg.inv(
                torch.as_tensor(camera_pose_data["extrinsic"], device=device, dtype=torch.float32)
            )
        camera_poses_7d = compute_camera_world_pose_7d(anchor_pose_7d, T_base_camera.reshape(-1, 4, 4))

        with open(renderer_cfg.camera_intrinsic_path, "r", encoding="utf-8") as f:
            intr_data = json.load(f)["intrinsic"]  # (fx, fy, ppx, ppy, height, width, depth_scale)
        d = renderer_cfg.render_downscale
        camera_intrinsic = np.eye(3)
        camera_intrinsic[0, 0] = intr_data[0] / d
        camera_intrinsic[1, 1] = intr_data[1] / d
        camera_intrinsic[0, 2] = intr_data[2] / d
        camera_intrinsic[1, 2] = intr_data[3] / d
        camera_height = int(intr_data[4] / d)
        camera_width = int(intr_data[5] / d)

        renderer_cfg.env_config.camera_intrinsic = tuple(map(tuple, camera_intrinsic))
        renderer_cfg.env_config.camera_width = camera_width
        renderer_cfg.env_config.camera_height = camera_height
    renderer = build_renderer(renderer_cfg, robot_config, device)

    Path(rendered_data_path).parent.mkdir(parents=True, exist_ok=True)

    robot_root_pose = (
        renderer.robot_helper.root_pose_7d.to(device)
        if robot_config.base_type == "fixed"
        else torch.tensor(PARKED_ROBOT_POSE_7D, device=device, dtype=torch.float32)
    )
    robot_qpos = renderer.robot_helper.init_qpos.to(device)  # chain-sized on a pruned robot
    if robot_config.base_type == "floating":
        robot_qpos = torch.cat([torch.zeros(6, device=device), robot_qpos])

    # sampled mode filters views by object visibility, which needs the intrinsic.
    visibility_intrinsic = (
        None if native_cameras else torch.as_tensor(camera_intrinsic, device=device, dtype=torch.float32)
    )
    logger.info(f"Render: camera_mode={renderer_cfg.camera_mode}, look_at_object={renderer_cfg.look_at_object}.")

    non_empty_scenes = 0
    for batch_scene_ids in track(
        batchify_tensor(scene_ids, renderer_cfg.num_scenes_per_batch),
        description="Rendering",
        total=math.ceil(len(scene_ids) / renderer_cfg.num_scenes_per_batch),
    ):
        batch_scene_descriptions = [scene_descriptions[scene_id] for scene_id in batch_scene_ids]
        num_scenes_b = len(batch_scene_descriptions)
        batch_camera_poses_7d = (
            None
            if native_cameras
            else _build_camera_poses(batch_scene_descriptions, camera_poses_7d, renderer_cfg, device)
        )
        num_views = len(renderer_cfg.env_config.cameras) if native_cameras else batch_camera_poses_7d.shape[1]
        batch_robot_qpos = (
            robot_qpos.expand(num_scenes_b, -1).unsqueeze(1)
            if robot_qpos_by_scene is None
            else torch.stack([robot_qpos_by_scene[sid] for sid in batch_scene_ids]).to(device)
        )
        batch_obs = renderer.render(
            scene_descriptions=batch_scene_descriptions,
            camera_poses=batch_camera_poses_7d,
            robot_root_poses=robot_root_pose,
            robot_qpos=batch_robot_qpos,
            object_poses=torch.tensor(
                [
                    (*s.meshes[s.target_actor_id].position, *s.meshes[s.target_actor_id].quaternion)
                    for s in batch_scene_descriptions
                ],
                device=device,
                dtype=torch.float32,
            )
            .unsqueeze(1)
            .expand(-1, batch_robot_qpos.shape[1], -1),
        )
        batch_pcd_obs = batch_obs["pointcloud"]
        camera_pose_frames = batch_obs["camera_poses"].reshape(num_scenes_b, num_views, batch_robot_qpos.shape[1], 7)

        # Renderer flattens (scene, view) into the leading dim; reshape back to (num_scenes, num_views, T, ...).
        batch_pcd_xyz = batch_pcd_obs["xyz"].reshape(num_scenes_b, num_views, *batch_pcd_obs["xyz"].shape[1:])
        batch_pcd_rgb = batch_pcd_obs["rgb"].reshape(num_scenes_b, num_views, *batch_pcd_obs["rgb"].shape[1:])
        batch_pcd_seg = batch_pcd_obs["seg"].reshape(num_scenes_b, num_views, *batch_pcd_obs["seg"].shape[1:])
        batch_seg_id = batch_pcd_obs["seg_id"].reshape(num_scenes_b, num_views, *batch_pcd_obs["seg_id"].shape[1:])
        images = batch_obs["sensor_data"]["base_camera"]["rgb"]
        images = images.reshape(num_scenes_b, num_views, *images.shape[1:])

        # sampled mode: keep every valid view per scene as that scene's view (T) axis; ragged -> save per scene.
        valid_view_indices = (
            [list(range(num_views)) for _ in batch_scene_ids]
            if renderer_cfg.camera_mode == "fixed"
            else _select_valid_view_indices(
                scene_descriptions=batch_scene_descriptions,
                camera_poses_7d=batch_camera_poses_7d,
                pcd_xyz=batch_pcd_xyz.numpy(force=True),
                intrinsic=visibility_intrinsic,
                image_size=(camera_width, camera_height),
                renderer_cfg=renderer_cfg,
            )
        )
        for s, scene_id in enumerate(batch_scene_ids):
            vi = valid_view_indices[s]
            if not vi:
                continue
            vt = torch.tensor(vi, dtype=torch.long)
            non_empty_scenes += bool(batch_pcd_xyz[s, vt].abs().sum() > 0)
            GraspKitData(
                scene_ids=[scene_id],
                pointcloud=batch_pcd_xyz[s, vt].transpose(0, 1).flatten(0, 1).unsqueeze(0).float(),
                pointcloud_rgb=batch_pcd_rgb[s, vt].transpose(0, 1).flatten(0, 1).unsqueeze(0).float(),
                pointcloud_segmentation=batch_pcd_seg[s, vt].transpose(0, 1).flatten(0, 1).unsqueeze(0).int(),
                target_actor_segmentation_id=batch_seg_id[s, vt[0], 0, 0].reshape(1).int(),
                camera_poses=camera_pose_frames[s, vt].transpose(0, 1).flatten(0, 1).unsqueeze(0).cpu().float(),
            ).save(rendered_data_path)
            merged = rendered_data_path.endswith(".h5")
            GraspKitData._write_h5(
                rendered_data_path if merged else str(Path(rendered_data_path) / f"{scene_id}.h5"),
                {scene_id: {"rgb": images[s, vt].transpose(0, 1).flatten(0, 1)}},
                group_per_scene=merged,
                append=False,
            )

    assert non_empty_scenes > 0, f"Render produced only all-zero pointclouds across {rendered_data_path}"

    write_completion_marker(rendered_data_path + RENDER_COMPLETE_MARKER)


if __name__ == "__main__":
    from graspkit.configs.experiment import cli_experiment

    exp_cfg: ExperimentConfig = cli_experiment()
    render_scene(exp_cfg)
