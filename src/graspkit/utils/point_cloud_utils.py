import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from robokit.xform.warp.torch_wrappers import transform_points

from graspkit.utils.tui_utils import log_once

BoundsLike = Union[
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    torch.Tensor,
]


def crop_and_resample_point_cloud(
    observation: Dict[str, torch.Tensor],
    bounds: BoundsLike,
    num_points: int,
) -> Dict[str, torch.Tensor]:
    """Crop to fixed `((xmin,xmax),(ymin,ymax),(zmin,zmax))` or per-sample `(B,3,2)` bounds, then resample."""
    xyzw = observation["xyzw"]
    b = torch.as_tensor(bounds, device=xyzw.device, dtype=xyzw.dtype)
    if b.dim() == 2:
        b = b.unsqueeze(0)  # (1, 3, 2) — broadcasts across B
    mins, maxs = b[..., 0].unsqueeze(1), b[..., 1].unsqueeze(1)  # (B|1, 1, 3)
    crop_mask = (xyzw[..., :3] >= mins).all(dim=-1) & (xyzw[..., :3] <= maxs).all(dim=-1) & (xyzw[..., 3] > 0)

    num_cropped_points = crop_mask.sum(dim=1)
    zero_points_mask = num_cropped_points == 0
    enough_points_mask = num_cropped_points >= num_points
    not_enough_points_mask = (~enough_points_mask) & (~zero_points_mask)

    resampled_indices = torch.zeros((xyzw.shape[0], num_points), dtype=torch.long, device=xyzw.device)

    # Handle "enough points" case (sample without replacement)
    if torch.any(enough_points_mask):
        weights = crop_mask[enough_points_mask].float()
        indices = torch.multinomial(weights, num_points, replacement=False)
        resampled_indices[enough_points_mask] = indices

    # Handle "not enough points" case (sample with replacement)
    if torch.any(not_enough_points_mask):
        log_once(
            "Not enough points for target object, using random indices with replacement",
            level=logging.WARNING,
        )
        weights = crop_mask[not_enough_points_mask].float()
        indices = torch.multinomial(weights, num_points, replacement=True)
        resampled_indices[not_enough_points_mask] = indices

    expanded_indices_xyz = resampled_indices.unsqueeze(-1).expand(-1, -1, 3)
    resampled_xyz = torch.gather(observation["xyzw"][..., :3], 1, expanded_indices_xyz)

    expanded_indices_rgb = resampled_indices.unsqueeze(-1).expand(-1, -1, 3)
    resampled_rgb = torch.gather(observation["rgb"][..., :3], 1, expanded_indices_rgb)

    expanded_indices_seg = resampled_indices.unsqueeze(-1).expand(-1, -1, 1)
    resampled_seg = torch.gather(observation["segmentation"][..., :1], 1, expanded_indices_seg)

    # Handle "zero points" case
    if torch.any(zero_points_mask):
        log_once(
            f"No points found in bounds {bounds} for sample, using all zero points",
            level=logging.WARNING,
        )
        resampled_xyz[zero_points_mask] = 0
        resampled_rgb[zero_points_mask] = 0
        resampled_seg[zero_points_mask] = 0

    return {"xyz": resampled_xyz, "rgb": resampled_rgb, "segmentation": resampled_seg}


def remove_point_cloud_segments_inplace(
    observation: Dict[str, torch.Tensor],
    seg_ids_to_remove: torch.Tensor,
    use_cpu: bool = False,
):
    """Zero out points with specified segmentation IDs in place, avoiding new tensor allocations to save memory.

    Args:
        observation: dictionary containing point cloud tensors. Will be modified in place.
        seg_ids_to_remove: IDs to remove. 1D tensor (all envs) or 2D tensor (per-env, shape [num_envs, num_ids]).
        use_cpu: if True, compute the segmentation mask on CPU to save GPU memory,
    """
    device = observation["segmentation"].device

    # Handle 1D tensor of ids by repeating for each env
    if seg_ids_to_remove.dim() == 1:
        num_envs = observation["segmentation"].shape[0]
        seg_ids_to_remove = seg_ids_to_remove.unsqueeze(0).repeat(num_envs, 1)

    if use_cpu and device.type != "cpu":
        # compute the mask on CPU to save memory
        cpu_seg = observation["segmentation"].cpu()
        cpu_ids_to_remove = seg_ids_to_remove.cpu()
        # Compare each point's seg id with the list of ids to remove for its env
        mask_to_remove_cpu = torch.any(cpu_seg == cpu_ids_to_remove.unsqueeze(1), dim=2)
        mask_to_remove = mask_to_remove_cpu.to(device)
    else:  # GPU or already on CPU
        seg = observation["segmentation"]
        mask_to_remove = torch.any(seg == seg_ids_to_remove.unsqueeze(1), dim=2)

    # modify the tensors in place
    observation["xyzw"][mask_to_remove] = 0
    observation["rgb"][mask_to_remove] = 0
    observation["segmentation"][mask_to_remove] = 0


def extract_point_cloud_segments_inplace(
    observation: Dict[str, torch.Tensor],
    seg_ids_to_keep: torch.Tensor,
    use_cpu: bool = False,
):
    """Zero out points outside the kept segmentation IDs in place, avoiding new tensor allocations to save memory.

    Args:
        observation: dictionary containing point cloud tensors. Will be modified in place.
        seg_ids_to_keep: IDs to keep. 1D tensor (all envs) or 2D tensor (per-env, shape [num_envs, num_ids]).
        use_cpu: if True, compute the mask on CPU to save GPU memory, then move it to GPU for the final op.
    """
    device = observation["segmentation"].device

    # Handle 1D tensor of ids by repeating for each env
    if seg_ids_to_keep.dim() == 1:
        num_envs = observation["segmentation"].shape[0]
        seg_ids_to_keep = seg_ids_to_keep.unsqueeze(0).repeat(num_envs, 1)

    if use_cpu and device.type != "cpu":
        # compute the mask on CPU to save memory
        cpu_seg = observation["segmentation"].cpu()
        cpu_ids_to_keep = seg_ids_to_keep.cpu()
        # Find points that are NOT in the keep list
        mask_to_keep_cpu = torch.any(cpu_seg == cpu_ids_to_keep.unsqueeze(1), dim=2)
        mask_to_remove = ~mask_to_keep_cpu
        mask_to_remove = mask_to_remove.to(device)
    else:  # GPU or already on CPU
        seg = observation["segmentation"]
        mask_to_keep = torch.any(seg == seg_ids_to_keep.unsqueeze(1), dim=2)
        mask_to_remove = ~mask_to_keep

    # modify the tensors in place
    observation["xyzw"][mask_to_remove] = 0
    observation["rgb"][mask_to_remove] = 0
    observation["segmentation"][mask_to_remove] = 0


def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project depth to a camera-frame point cloud; `intrinsics`=[fx,fy,ppx,ppy,h,w,depth_scale] (RealSense)."""
    fx, fy, ppx, ppy, h, w, depth_scale = intrinsics
    h, w = int(h), int(w)
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth.astype(np.float32) * depth_scale
    x = (xs - ppx) * z / fx
    y = (ys - ppy) * z / fy
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = z.ravel() > 0
    return points[valid].astype(np.float32)


def masked_depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Back-project only depth pixels selected by an aligned image mask."""
    assert mask.shape == depth.shape
    masked_depth = depth.copy()
    masked_depth[~mask.astype(bool)] = 0
    return depth_to_point_cloud(masked_depth, intrinsics)


def process_point_cloud(
    points: np.ndarray,
    transform: np.ndarray,
    workspace_center: Optional[Tuple[float, float, float]] = None,
    sample_size: int = 4096,
    crop_half_extent: float = 0.15,
    segment_distance_threshold: float = 0.008,
    z_filter_min_offset: float = 0.0,
) -> np.ndarray:
    """Map points to the robot base frame, crop near `workspace_center` if set, drop table/outliers, cluster, sample."""
    import open3d as o3d

    pts_robot = transform_points(torch.from_numpy(points), torch.from_numpy(transform)).numpy()
    if workspace_center is not None:
        x_min, x_max = workspace_center[0] - crop_half_extent, workspace_center[0] + crop_half_extent
        y_min, y_max = workspace_center[1] - crop_half_extent, workspace_center[1] + crop_half_extent
        mask = (
            (pts_robot[:, 0] > x_min)
            & (pts_robot[:, 0] < x_max)
            & (pts_robot[:, 1] > y_min)
            & (pts_robot[:, 1] < y_max)
        )
        pts_robot = pts_robot[mask]
    if pts_robot.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_robot)
    pcd_down = pcd.voxel_down_sample(0.005)
    _, inliers = pcd_down.segment_plane(distance_threshold=segment_distance_threshold, ransac_n=3, num_iterations=1000)
    pcd_no_table = pcd_down.select_by_index(inliers, invert=True)
    pcd_no_table, _ = pcd_no_table.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    labels = np.array(pcd_no_table.cluster_dbscan(eps=0.01, min_points=15))
    pts = np.asarray(pcd_no_table.points)[labels != -1]
    if z_filter_min_offset > 0.0 and pts.shape[0] > 0:
        pts = pts[pts[:, 2] > pts[:, 2].min() + z_filter_min_offset]

    if len(pts) > sample_size:
        indices = np.random.choice(pts.shape[0], sample_size, replace=False)
        pts = pts[indices]
    return np.array(pts, dtype=np.float32)
