"""Optimize xHand grasps for a YCB object and visualize them in Viser."""

import time

import torch
import trimesh
import viser
import yourdfpy
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_quaternion
from viser.extras import ViserUrdf

from graspkit.assets.objects import dex_diffuser
from graspkit.configs.robot.xhand import xhand_right
from graspkit.constants import OBJECT_COLOR, ROBOT_COLOR
from graspkit.stages.grasp_optim import GraspOptimConfig, MeshDescription, SceneDescription, WarpGraspOptimizer

OBJECT = "mustard_bottle"
YCB_ROOT = dex_diffuser.DATA_DIR / "processed/ycb"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = GraspOptimConfig(optimizer_backend="warp_gd", num_contact_points=5, scene_geom_backend="mesh")
    optimizer = WarpGraspOptimizer(config, xhand_right, device)

    # --- optimize --- one scene, one target mesh at the origin (no table)
    collision_glb = str(YCB_ROOT / OBJECT / "collision.glb")
    scene = SceneDescription(
        scene_id=OBJECT,
        target_actor_id=OBJECT,
        meshes={OBJECT: MeshDescription(actor_id=OBJECT, collision_mesh_path=collision_glb, scales=(1.0, 1.0, 1.0))},
    )
    final_variables = optimizer.optimize([scene], num_samples_per_scene=8)

    # --- viewer --- slide through the optimized grasps
    # each variable is [xyz(3), euler_rxyz(3), hand_q(rest)]; euler -> quat for viser
    base, q = final_variables[0][:, :6], final_variables[0][:, 6:]
    base_quat = matrix_to_quaternion(euler_angles_to_matrix(base[:, 3:], axes="rxyz"))

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=0.5, height=0.5)
    mesh = trimesh.load(collision_glb, process=False, force="mesh")
    server.scene.add_mesh_simple("/object", mesh.vertices, mesh.faces, color=OBJECT_COLOR, flat_shading=False)
    robot_base = server.scene.add_frame("/robot_base", show_axes=False)
    urdf_vis = ViserUrdf(
        server,
        yourdfpy.URDF.load(xhand_right.urdf_or_mjcf_path),
        root_node_name="/robot_base",
        mesh_color_override=ROBOT_COLOR,
    )
    sample_slider = server.gui.add_slider("Sample", 0, base.shape[0] - 1, 1, 0)

    def show(_=None):
        idx = int(sample_slider.value)
        with server.atomic():
            robot_base.position = base[idx, :3].numpy(force=True)
            robot_base.wxyz = base_quat[idx].numpy(force=True)
            urdf_vis.update_cfg(q[idx].numpy(force=True))

    sample_slider.on_update(show)
    show()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
