"""
python scripts/stages/asset_prep/process_dexgraspnet.py
"""

import glob
import os

import h5py
import numpy as np
import torch
import trimesh
from rich.progress import track
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_euler_angles

from graspkit.configs.robot import build_robot_config
from graspkit.constants import DATA_ROOT
from graspkit.robot.grasp_robot import GraspRobot

if __name__ == "__main__":
    data_dir = str(DATA_ROOT / "dexgraspnet")
    data_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    mesh_dir = str(DATA_ROOT / "object_data/dexgraspnet/meshdata")

    output_action_path = str(DATA_ROOT / "optim_data/dexgraspnet_grasps/dexgraspnet.h5")
    output_render_path = str(DATA_ROOT / "render_data/pointcloud/dexgraspnet/mesh_sample.h5")
    os.makedirs(os.path.dirname(output_action_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_render_path), exist_ok=True)

    robot_config = build_robot_config("shadow_right")
    robot_helper = GraspRobot.from_config(robot_config)

    with h5py.File(output_action_path, mode="w") as h5_action_file:
        with h5py.File(output_render_path, mode="w") as h5_render_file:
            for data_path in track(data_paths, description="Processing DexGraspNet data..."):
                obj_id = os.path.basename(data_path).split(".")[0]
                action_data = np.load(data_path, allow_pickle=True)
                full_qpos = torch.tensor([list(data["qpos"].values()) for data in action_data])
                root_xyz = full_qpos[:, -3:]
                root_rpy_sxyz = full_qpos[:, -6:-3]
                root_rpy_rxyz = matrix_to_euler_angles(euler_angles_to_matrix(root_rpy_sxyz, "sxyz"), "rxyz")
                qpos = full_qpos[:, :-6]
                joint_names = list(action_data[0]["qpos"].keys())[:-6]
                qpos_reindexed = robot_helper._reindex_joint_values(qpos, joint_names)
                qpos_ours = torch.cat([root_xyz, root_rpy_rxyz, qpos_reindexed], dim=-1)

                mesh_path = os.path.join(mesh_dir, f"{obj_id}/coacd/decomposed.obj")
                scales = [data["scale"] for data in action_data]
                unique_scales = list(set(scales))

                for unique_scale in unique_scales:
                    mesh = trimesh.load(mesh_path)
                    mesh.apply_scale(unique_scale)
                    pts = mesh.sample(4096)
                    scale_idx = [i for i, scale in enumerate(scales) if scale == unique_scale]

                    scene_id = f"{obj_id}_{unique_scale}"
                    if scene_id in h5_action_file:
                        del h5_action_file[scene_id]
                    if scene_id in h5_render_file:
                        del h5_render_file[scene_id]
                    scene_action_group = h5_action_file.create_group(scene_id)
                    scene_action_group.create_dataset("grasp_qpos", data=qpos_ours[scale_idx])
                    scene_action_group.create_dataset("pre_grasp_qpos", data=qpos_ours[scale_idx])
                    scene_action_group.create_dataset("post_grasp_qpos", data=qpos_ours[scale_idx])
                    scene_action_group.create_dataset("additional_post_grasp_qpos", data=qpos_ours[scale_idx])

                    scene_render_group = h5_render_file.create_group(scene_id)
                    scene_render_group.create_dataset("pointcloud", data=pts)
                    scene_render_group.create_dataset("pointcloud_rgb", data=pts)
                    scene_render_group.create_dataset("pointcloud_segmentation", data=np.zeros((pts.shape[0], 1)))
                    scene_render_group.create_dataset("target_actor_segmentation_id", data=0.0)

            h5_render_file.flush()

        h5_action_file.flush()
