"""Optimize Sharpa grasps for a tabletop object (floating base + table) and visualize in Viser."""

import time

import torch
import trimesh
import viser
import yourdfpy
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_quaternion
from viser.extras import ViserUrdf

from graspkit.assets.objects import dex_diffuser
from graspkit.configs.robot.sharpa import sharpa_right
from graspkit.constants import OBJECT_COLOR, ROBOT_COLOR, TABLE_COLOR
from graspkit.stages.grasp_optim import (
    BoxDescription,
    GraspOptimConfig,
    MeshDescription,
    SceneDescription,
    WarpGraspOptimizer,
)

OBJECT = "mustard_bottle"
YCB_ROOT = dex_diffuser.DATA_DIR / "processed/ycb"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = GraspOptimConfig(
        optimizer_backend="warp_gd",
        num_contact_points=5,
        scene_geom_backend="mesh",
        w_angle_z=100.0,
        w_spen=20.0,
        spen_threshold=0.0,
    )
    optimizer = WarpGraspOptimizer(config, sharpa_right, device)

    collision_glb = str(YCB_ROOT / OBJECT / "collision.glb")
    obj = MeshDescription(actor_id=OBJECT, collision_mesh_path=collision_glb, scales=(1.0, 1.0, 1.0))
    obj.position = (0.0, 0.0, -obj.get_bounds()[2][0])
    table = BoxDescription(actor_id="table", size=(0.3, 0.3, 0.1), position=(0.0, 0.0, -0.05))
    scene = SceneDescription(scene_id=OBJECT, target_actor_id=OBJECT, boxes={"table": table}, meshes={OBJECT: obj})
    final_variables = optimizer.optimize([scene], num_samples_per_scene=8)

    # --- viewer --- slide through the optimized grasps
    base, q = final_variables[0][:, :6], final_variables[0][:, 6:]
    base_quat = matrix_to_quaternion(euler_angles_to_matrix(base[:, 3:], axes="rxyz"))

    server = viser.ViserServer()
    server.scene.add_box(
        "/table", color=TABLE_COLOR, dimensions=table.size, position=table.position, wxyz=table.quaternion
    )
    mesh = trimesh.load(obj.collision_mesh_path, process=False, force="mesh")
    server.scene.add_mesh_simple(
        "/object",
        mesh.vertices,
        mesh.faces,
        color=OBJECT_COLOR,
        flat_shading=False,
        scale=obj.scales,
        position=obj.position,
        wxyz=obj.quaternion,
    )
    robot_base = server.scene.add_frame("/robot_base", show_axes=False)
    urdf_vis = ViserUrdf(
        server,
        yourdfpy.URDF.load(sharpa_right.urdf_or_mjcf_path),
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
