"""Perception augmentation for camera object clouds (mask errors in each view's image plane), plus thinning down to 10%.

Measured on RoboDojo gate clouds:

- views: the head (f 286 px, 0.52-0.65 m) + wrist looks; all wrist views dropped with p 0.25, a second wrist view
  dropped with p 0.5;
- masks (SAM 3): eroded k px, P(k = 0, 1, 2, 3) = 0.25, 0.45, 0.2, 0.1; masks < 1500 px also lose 0-20% of their
  boundary pixels; p 0.15 a 1 px dilation onto the table (neighbour bleed is negligible);
- depth: sim depth has no noise; for RGB / MoGe robustness (p `rgb_prob`), per view and object
  d' = t_mean + b + a (t - t_mean) + r along each camera ray, b ~ N(0, 8 mm) clipped +-2.5 cm, a ~ U(0.45, 1) head /
  U(0.6, 1.2) wrist, r smooth (sigma 3 mm, ~2 cm correlation), and 20% of the outline pixels dropped;
- misregistration (Isaac 6): a copy of one wrist view translated U(1, 4) cm, rotated <= 3 deg about its camera; p 0.05
  (was 0.2 until the perception agent's GRASP_VIEWS_SYNC=1 removed most of it at the source);
- cuts: points below table + 2 mm removed; below table + 4 mm too unless the object is flat;
- density: the new gate pipeline keeps raw camera clouds (GRASP_VIEWS_VOXEL=0); voxel 3 mm with p 0.3 for the old one.
"""

import numpy as np
from robokit.xform.numpy import axis_angle_to_matrix, quaternion_to_matrix
from scipy import ndimage
from scipy.spatial import cKDTree

F_PX = 286.311  # RoboDojo cam_head focal length (px); the renders use it for every view
ERODE_P = (0.25, 0.45, 0.2, 0.1)


def _smooth_noise(pts: np.ndarray, rng: np.random.Generator, sigma: float = 0.003, corr: float = 0.02) -> np.ndarray:
    """Spatially smooth scalar noise at pts: Gaussian-kernel mix of random anchors (std ~sigma, ~corr length)."""
    if len(pts) == 0:
        return np.zeros(0)
    k = 24
    anchors = pts[rng.integers(len(pts), size=k)] + rng.normal(scale=corr, size=(k, 3))
    w = np.exp(-np.sum((pts[:, None] - anchors[None]) ** 2, -1) / (2 * corr**2))
    r = w @ rng.normal(size=k)
    s = r.std()
    return r / s * sigma if s > 1e-9 else r


def augment_view(
    pts: np.ndarray,  # (N, 3) world, this view's points (target + neighbours; table/robot removed)
    seg: np.ndarray,  # (N,) segment ids
    target_id: int,
    pose7: np.ndarray,  # (7,) SAPIEN camera pose (x forward, y left, z up), p + q wxyz
    rng: np.random.Generator,
    head: bool,
    rgb: bool,
    table_z: float = 0.0,
    dilate_prob: float = 0.15,
) -> np.ndarray:
    """The target's points in this view after mask errors, depth errors (`rgb`) and the table cuts, (M, 3)."""
    valid = np.linalg.norm(pts, axis=1) > 1e-6
    pts, seg = pts[valid], seg[valid]
    tgt = seg == target_id
    if tgt.sum() < 8:
        return pts[tgt]
    R = quaternion_to_matrix(pose7[3:7])
    cam = (pts - pose7[:3]) @ R
    depth = np.maximum(cam[:, 0], 1e-3)
    uv = np.stack([-cam[:, 1] / depth, -cam[:, 2] / depth], 1) * F_PX  # pixels
    spacing = float(np.median(cKDTree(uv[tgt]).query(uv[tgt], k=2)[0][:, 1]))
    cell = max(spacing, 1.0)
    lo = np.floor(uv[tgt].min(0) / cell) - 4
    cells = np.floor(uv / cell - lo).astype(int)
    shape = tuple(cells[tgt].max(0) + 5)
    inb = np.all((cells >= 0) & (cells < np.array(shape)), 1)
    ci = np.clip(cells, 0, np.array(shape) - 1)
    mask = np.zeros(shape, bool)
    mask[cells[tgt, 0], cells[tgt, 1]] = True
    mask = ndimage.binary_closing(mask, iterations=2, border_value=0) | mask
    mask = ndimage.binary_fill_holes(mask)
    area_px = mask.sum() * cell * cell
    edge_px = ndimage.distance_transform_edt(mask)[ci[:, 0], ci[:, 1]] * cell  # distance to the mask's edge (px)
    keep = tgt.copy()
    k = rng.choice(4, p=ERODE_P)
    if k:
        keep &= edge_px > k
    if area_px < 1500:  # small masks: 0-20% of the boundary pixels too
        boundary = keep & (edge_px < k + 2.0)
        keep &= ~(boundary & (rng.random(len(keep)) < rng.uniform(0.0, 0.2)))
    if rgb:  # RGB / MoGe depth: 20% of the outline dropped
        outline = keep & (edge_px < k + 2.0)
        keep &= ~(outline & (rng.random(len(keep)) < 0.2))
    if keep.sum() < 8:
        keep = tgt.copy()
    extra = []
    if rng.random() < dilate_prob:  # 1 px onto the table: ring cells with no object see the table top
        ring = ndimage.binary_dilation(mask, iterations=max(1, int(round(1.0 / cell)))) & ~mask
        occupied = np.zeros(shape, bool)
        occupied[cells[inb, 0], cells[inb, 1]] = True
        free = np.argwhere(ring & ~occupied)
        if len(free):
            u = (free + lo + rng.random(free.shape)) * cell / F_PX
            d_w = np.stack([np.ones(len(u)), -u[:, 0], -u[:, 1]], 1) @ R.T
            t = (table_z - pose7[2]) / np.where(np.abs(d_w[:, 2]) > 1e-6, d_w[:, 2], -1e-6)
            ok = (t > 0) & (t < 3.0)
            extra.append(pose7[:3] + t[ok, None] * d_w[ok])
    out = pts[keep]
    if extra:
        out = np.concatenate([out, *extra])
    if rgb and len(out):  # d' = t_mean + b + a (t - t_mean) + r along each camera ray
        ray = out - pose7[:3]
        t = np.linalg.norm(ray, axis=1)
        tm = t.mean()
        b = float(np.clip(rng.normal(scale=0.008), -0.025, 0.025))
        a = rng.uniform(0.45, 1.0) if head else rng.uniform(0.6, 1.2)
        t2 = tm + b + a * (t - tm) + _smooth_noise(out, rng)
        out = pose7[:3] + ray / np.maximum(t, 1e-6)[:, None] * t2[:, None]
    z_all = pts[tgt, 2]
    flat = z_all.max() - table_z < 0.015
    out = out[out[:, 2] > table_z + (0.002 if flat else 0.004)]
    return out.astype(np.float32)


def misregister(points: np.ndarray, pose7: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A copy of a wrist view's points translated U(1, 4) cm and rotated <= 3 deg about the wrist camera."""
    if len(points) == 0:
        return points
    d = rng.normal(size=3)
    t = d / np.linalg.norm(d) * rng.uniform(0.01, 0.04)
    ax = rng.normal(size=3)
    Rr = axis_angle_to_matrix(ax / np.linalg.norm(ax) * np.radians(rng.uniform(0, 3)))
    return ((points - pose7[:3]) @ Rr.T + pose7[:3] + t).astype(np.float32)


def thin(points: np.ndarray, rng: np.random.Generator, keep_min: float, voxel: float) -> np.ndarray:
    """p 0.3 voxel dedup at `voxel` (old gate pipeline: 3 mm); p 0.2 a keep fraction log-uniform in [keep_min, 1]."""
    if len(points) == 0:
        return points
    if voxel > 0 and rng.random() < 0.3:
        _, first = np.unique(
            np.floor((points + rng.random(3) * voxel) / voxel).astype(np.int64), axis=0, return_index=True
        )
        points = points[np.sort(first)]
    if rng.random() < 0.2:
        k = max(8, int(round(len(points) * float(np.exp(rng.uniform(np.log(keep_min), 0.0))))))
        points = points[rng.permutation(len(points))[:k]]
    return points


def fuse(
    pcd_views: np.ndarray,  # (V, N, 3) world; view 0 = the head camera
    seg_views: np.ndarray,  # (V, N) segment ids
    target_id: int,
    cams: np.ndarray,  # (V, 7) camera poses
    rng: np.random.Generator,
    aug: bool,
    max_wrist: int = 2,
    drop_wrist_prob: float = 0.25,
    rgb_prob: float = 0.3,
    keep_min: float = 0.1,
    voxel: float = 0.003,
    misregister_prob: float = 0.05,
    view_noise: float = 0.0,
) -> np.ndarray:
    """One object cloud: head + up to `max_wrist` random wrist views, through the perception errors above when `aug`."""
    others = rng.permutation(np.arange(1, len(pcd_views)))
    wrist = list(others[:max_wrist])
    if aug:
        if rng.random() < drop_wrist_prob:
            wrist = []
        elif len(wrist) > 1 and rng.random() < 0.5:
            wrist = wrist[:1]
    views = [0, *wrist]
    rgb = aug and rng.random() < rgb_prob
    clouds = []
    for v in views:
        if aug:
            c = augment_view(pcd_views[v], seg_views[v], target_id, cams[v], rng, head=v == 0, rgb=rgb)
        else:
            c = pcd_views[v][(seg_views[v] == target_id) & (np.linalg.norm(pcd_views[v], axis=1) > 1e-6)]
        if view_noise > 0 and len(c):  # small per-view registration jitter
            Rj = axis_angle_to_matrix(rng.normal(size=3) * view_noise * 10)
            m = c.mean(0)
            c = (c - m) @ Rj.T + m + rng.normal(size=3) * view_noise
        clouds.append(c)
    if aug and len(views) > 1 and rng.random() < misregister_prob:
        w = int(rng.integers(1, len(views)))
        clouds.append(misregister(clouds[w], cams[views[w]], rng))
    pts = np.concatenate(clouds).astype(np.float32)
    return thin(pts, rng, keep_min, voxel) if aug else pts
