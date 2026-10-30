from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float
from robokit.utils.ee_coord_utils import HandCoordinateSpec
from robokit.xform.utils import coord_conversion
from robokit.xform.warp.torch_wrappers import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    pose7_to_tf_mat,
    rot_tl_to_tf_mat,
    tf_mat_to_pose7,
)

OPENGL = HandCoordinateSpec(palm_forward_axis="-z", four_fingers_up_axis="y")  # x right, y up, z back


@torch.enable_grad()
def matrix_to_continuous_euler_angles(
    rot_mat: Float[torch.Tensor, "... t 3 3"],
    axes: str = "rxyz",
    init_angles: Optional[Float[torch.Tensor, "... 3"]] = None,
    num_iters: int = 50,
) -> Float[torch.Tensor, "... t 3"]:
    rot_mat = rot_mat.clone().detach()
    euler_angles = torch.zeros_like(rot_mat[..., 0])

    if init_angles is None:
        euler_angles[..., 0, :] = matrix_to_euler_angles(rot_mat[..., 0, :3, :3], axes)
    else:
        euler_angles[..., 0, :] = init_angles

    for frame_idx in range(1, rot_mat.shape[-3]):
        opt_angles = euler_angles[..., frame_idx - 1, :].clone().detach()
        init_angles = opt_angles.clone()
        opt_angles.requires_grad_(True)
        optimizer = torch.optim.Adam([opt_angles], lr=0.01)
        for _ in range(num_iters):
            rot_mat_opt = euler_angles_to_matrix(opt_angles, axes)
            traces = (
                (rot_mat[..., frame_idx, :, :] @ rot_mat_opt.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            )
            dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + 1e-7, 1 - 1e-7)).mean()
            reg_loss = F.mse_loss(opt_angles, init_angles)
            loss = dists + reg_loss * 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        euler_angles[..., frame_idx, :] = opt_angles.clone().detach()

    return euler_angles


def compute_camera_world_pose_7d(
    robot_pose_7d: Float[torch.Tensor, "... 7"],
    T_base_camera: Float[torch.Tensor, "... 4 4"],
) -> Float[torch.Tensor, "... 7"]:
    """Combine a robot base pose with a base-frame camera pose (OpenCV, from calib JSONs) into a SAPIEN world pose."""
    robot_pose_mat = pose7_to_tf_mat(robot_pose_7d)
    sapien2opencv = coord_conversion("sapien", "opencv", return_tensors="pt").to(robot_pose_mat.device)
    sapien2opencv = rot_tl_to_tf_mat(rot_mat=sapien2opencv)
    return tf_mat_to_pose7(robot_pose_mat @ T_base_camera @ sapien2opencv)
