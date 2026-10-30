"""Offline viser viz of robot URDF + calibrated camera frame (world_cam = init_base_pose @ T_base_camera).

Usage:
    uv run scripts/tools/visualize_camera_extrinsic.py --experiment-name g1_inspire_right_wbc_dex1b_v0_5_mini
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf

from graspkit.configs.experiment import build_experiment_config
from graspkit.constants import ROBOT_COLOR
from graspkit.utils.xform_utils import compute_camera_world_pose_7d


@dataclass
class Args:
    experiment_name: str
    axis_length: float = 0.15
    """Length of the camera/base frame axes (meters)."""


def main(args: Args) -> None:
    exp = build_experiment_config(args.experiment_name)
    rc = exp.traj_builder.fixed_base_robot_config
    assert rc is not None, f"{args.experiment_name} has no fixed_base_robot_config"

    # Camera world pose = init_base_pose @ T_base_camera (matches render_data.py).
    with open(exp.renderer.camera_poses_path) as f:
        data = json.load(f)
    if "T_base_camera" in data:
        t_base_camera = torch.tensor(data["T_base_camera"], dtype=torch.float32)
    else:
        t_base_camera = torch.linalg.inv(torch.tensor(data["extrinsic"], dtype=torch.float32))
    base_pose = torch.tensor((*rc.init_position, *rc.init_quaternion), dtype=torch.float32)
    cam_pose = compute_camera_world_pose_7d(base_pose, t_base_camera.unsqueeze(0))[0].numpy()  # (7,) xyz+wxyz

    load_kwargs = {"build_scene_graph": True, "load_meshes": True}
    if rc.mesh_dir is not None:
        load_kwargs["mesh_dir"] = rc.mesh_dir
    urdf = yourdfpy.URDF.load(rc.urdf_or_mjcf_path, **load_kwargs)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    # Robot at its init base pose; URDF at init joints if the count matches the actuated DOFs.
    server.scene.add_frame(
        "/robot",
        show_axes=True,
        axes_length=args.axis_length,
        position=np.asarray(rc.init_position),
        wxyz=np.asarray(rc.init_quaternion),
    )
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot", mesh_color_override=ROBOT_COLOR)
    n_act = len(urdf.actuated_joints)
    if rc.init_qpos is not None and len(rc.init_qpos) == n_act:
        urdf_vis.update_cfg(np.asarray(rc.init_qpos, dtype=np.float32))
    else:
        urdf_vis.update_cfg(np.zeros(n_act, dtype=np.float32))

    server.scene.add_frame(
        "/camera",
        show_axes=True,
        axes_length=args.axis_length,
        position=cam_pose[:3],
        wxyz=cam_pose[3:],
    )
    print(f"camera world pose (xyz, wxyz): {cam_pose.round(4).tolist()}")
    print(f"viser ready (base={tuple(rc.init_position)}). Ctrl-C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
