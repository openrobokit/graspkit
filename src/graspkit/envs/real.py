import time
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from robokit.xform.warp.torch_wrappers import transform_points
from torch.types import Device

from graspkit.configs.real_env_config import RealEnvConfig
from graspkit.configs.robot import build_robot_config
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs.base import BaseEnv
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.point_cloud_utils import masked_depth_to_point_cloud


class RealEnv(BaseEnv):
    """Hardware-backed grasp env with config-selected arm/hand/camera; same primitives as MsEnv, no evaluate()."""

    def __init__(
        self,
        real_env_config: RealEnvConfig,
        device: Device,
        t_robot_cam: np.ndarray,
        workspace_center: Optional[Tuple[float, float, float]] = None,
        pc_sample_size: int = 4096,
    ):
        super().__init__()
        self.robot_config = build_robot_config(real_env_config.robot_config_name)
        self.device = device
        self.robot_helper = GraspRobot.from_config(self.robot_config, device=device)
        # config-derived arm/hand DOF split (order- and count-agnostic; handles 6- or 7-dof arms)
        self.arm_idx = self.robot_helper.arm_joint_indices.cpu().numpy()
        self.hand_idx = self.robot_helper.hand_joint_indices.cpu().numpy()
        self.t_robot_cam = np.asarray(t_robot_cam, dtype=np.float32)
        self.workspace_center = workspace_center or real_env_config.workspace_center
        self.hand_reindex = real_env_config.hand_reindex  # ours -> real hand order; None = identity
        self.control_hz = real_env_config.control_hz
        self.pc_sample_size = pc_sample_size
        self.pc_filter_radius = real_env_config.pc_filter_radius
        self.depth_frames = real_env_config.depth_frames
        assert self.depth_frames > 0
        self.arm = real_env_config.make_arm()  # factories return already-started hardware
        self.hand = real_env_config.make_hand()
        self.camera = real_env_config.make_camera()

    def _hand_to_real(self, hand: np.ndarray) -> np.ndarray:
        return hand if self.hand_reindex is None else hand[..., self.hand_reindex]

    def _hand_to_ours(self, hand: np.ndarray) -> np.ndarray:
        return hand if self.hand_reindex is None else hand[..., np.argsort(self.hand_reindex)]

    def reset(
        self,
        scene: Optional[SceneDescription] = None,
        init_qpos: Optional[torch.Tensor] = None,
        root_pose: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
    ):
        q = (init_qpos.squeeze(0) if init_qpos.dim() == 2 else init_qpos).cpu().numpy()
        self.hand.set_qpos(self._hand_to_real(q[self.hand_idx]))
        self.arm.set_qpos(q[self.arm_idx], wait=True)

    def get_qpos(self) -> torch.Tensor:
        q = self.robot_helper.init_qpos.clone()  # chain joints outside arm+hand stay frozen at init
        q[self.arm_idx] = torch.from_numpy(self.arm.get_qpos().astype(np.float32)).to(self.device)
        q[self.hand_idx] = torch.from_numpy(self._hand_to_ours(self.hand.get_qpos()).astype(np.float32)).to(self.device)
        return q.unsqueeze(0)

    def get_qpos_cpu(self) -> torch.Tensor:
        """CUDA-free `get_qpos` for the visualizer thread, whose GPU ops would kill the planner's CUDA graph capture."""
        q = self.robot_helper.init_qpos.cpu().numpy().copy()
        q[self.arm_idx] = self.arm.get_qpos().astype(np.float32)
        q[self.hand_idx] = self._hand_to_ours(self.hand.get_qpos()).astype(np.float32)
        return torch.from_numpy(q).unsqueeze(0)

    def set_qpos_traj(self, traj_qpos, qpos_callback=None) -> dict[str, torch.Tensor]:
        """One firmware-interpolated arm trajectory per call; the hand ramps to the chunk's last frame."""
        traj = traj_qpos.cpu().numpy()
        self.hand.set_qpos(self._hand_to_real(traj[-1, self.hand_idx]))
        self.arm.set_qpos_traj(traj[:, self.arm_idx], self.control_hz)
        if qpos_callback is not None:
            # Both: the caller picks which to draw; their gap is tracking error and settling, not calibration.
            qpos_callback(traj_qpos.shape[0] - 1, self.get_qpos()[0], traj_qpos[-1])
        return self.get_info()

    def step_qpos(self, q: torch.Tensor):
        q = (q.squeeze(0) if q.dim() == 2 else q).cpu().numpy()
        self.arm.set_qpos(q[self.arm_idx])
        self.hand.set_qpos(self._hand_to_real(q[self.hand_idx]))
        time.sleep(1 / self.control_hz)

    def get_info(self) -> dict[str, torch.Tensor]:
        return {}

    def get_rgb(self) -> np.ndarray:
        # A single ZED frame is noisy enough to look like a calibration error; median a few.
        depths = np.stack([self.camera.get_depth() for _ in range(self.depth_frames)])
        valid = np.isfinite(depths) & (depths > 0)
        self._captured_depth = np.ma.median(np.ma.array(depths, mask=~valid), axis=0).filled(0).astype(np.float32)
        self._captured_rgb = self.camera.get_color()[..., ::-1]
        return self._captured_rgb

    def get_pointcloud(self, mask: np.ndarray) -> torch.Tensor:
        assert mask.shape == self._captured_depth.shape == self._captured_rgb.shape[:2]
        points = masked_depth_to_point_cloud(self._captured_depth, self.camera.get_intrinsics(), mask)
        if self.pc_filter_radius > 0.0:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            cloud, _ = cloud.remove_radius_outlier(nb_points=16, radius=self.pc_filter_radius)
            filtered = np.asarray(cloud.points, dtype=np.float32)
            print(f"3D radius filter: {len(points)} -> {len(filtered)} points ({self.pc_filter_radius:.3f} m)")
            points = filtered
        pts = torch.from_numpy(points).to(self.device)
        pts = transform_points(pts, torch.from_numpy(self.t_robot_cam).to(self.device))
        center = torch.tensor(self.workspace_center, device=self.device)
        keep = (torch.abs(pts[:, :2] - center[:2]) < 0.15).all(dim=1)
        pts = pts[keep]
        if pts.shape[0] > self.pc_sample_size:
            pts = pts[torch.randperm(pts.shape[0], device=self.device)[: self.pc_sample_size]]
        return transform_points(pts, self.robot_helper.root_pose_mat).unsqueeze(0)

    def get_observed_cloud(self, max_points: int = 60000) -> Tuple[np.ndarray, np.ndarray]:
        """Fresh colored robot-frame cloud for eyeballing calibration; no crop or filter, so the arm and hand stay."""
        depths = np.stack([self.camera.get_depth() for _ in range(self.depth_frames)])
        rgb = self.camera.get_color()[..., ::-1]
        ok = np.isfinite(depths) & (depths > 0)
        depth = np.ma.median(np.ma.array(depths, mask=~ok), axis=0).filled(0).astype(np.float32)
        valid = depth > 0.05
        # depth_to_point_cloud keeps exactly the z>0 pixels (`valid`), so the same flat mask selects matching colors.
        points = masked_depth_to_point_cloud(depth, self.camera.get_intrinsics(), valid)
        colors = rgb.reshape(-1, 3)[valid.reshape(-1)]
        assert points.shape[0] == colors.shape[0], f"{points.shape} vs {colors.shape}"
        if points.shape[0] > max_points:
            keep = np.random.default_rng(0).choice(points.shape[0], max_points, replace=False)
            points, colors = points[keep], colors[keep]
        pts = transform_points(
            torch.from_numpy(points).to(self.device), torch.from_numpy(self.t_robot_cam).to(self.device)
        )
        pts = transform_points(pts, self.robot_helper.root_pose_mat)
        return pts.cpu().numpy().astype(np.float32), colors.astype(np.uint8)

    def evaluate(self, batch: GraspKitData, record: bool = False):
        raise NotImplementedError(
            "RealEnv has no ground-truth success: execute a trajectory by looping step_qpos and "
            "label success out-of-band (e.g. Gemini/manual)."
        )

    def close(self):
        self.camera.stop()
