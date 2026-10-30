"""Click a hand mesh in viser to add contact points (or near one to delete); exports link-local contact_points.json.

Usage:
    uv run scripts/tools/create_contact_points.py --urdf-path <hand.urdf>
    uv run scripts/tools/create_contact_points.py --urdf-path <hand.urdf> \
        --contact-points-path <existing_contact_points.json>
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import tyro
import viser
import yourdfpy
from robokit.robo.robot import Robot

from graspkit.constants import ROBOT_COLOR


@dataclass
class Args:
    urdf_path: str
    mesh_dir: Optional[str] = None
    base_link_name: Optional[str] = None
    contact_points_path: Optional[str] = None
    """Existing file to seed from AND the export target. Defaults to <mesh_dir>/contact_points.json."""
    point_size: float = 0.001
    remove_radius: float = 0.001
    update_hz: float = 30.0


def main(args: Args) -> None:
    urdf_path = Path(args.urdf_path)
    mesh_dir = args.mesh_dir or str(urdf_path.parent)
    export_path = Path(args.contact_points_path or (Path(mesh_dir) / "contact_points.json"))

    robot = Robot.load(str(urdf_path), load_meshes=True, mesh_dir=mesh_dir, base_link_name=args.base_link_name)
    link_names = robot.spec.link_names
    # Link-local collision meshes for the links that actually have geometry.
    local_meshes = {}
    for link_name in link_names:
        mesh = robot.spec.get_link_mesh(link_name, mode="collision", return_mesh=True)
        if len(mesh.faces) > 0:
            # A box or cylinder collision comes back as a trimesh primitive, which refuses to be posed.
            local_meshes[link_name] = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())

    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        filename_handler=lambda fname: yourdfpy.filename_handler_magic(fname, dir=mesh_dir),
    )

    # Seed points from an existing file (link-local), keeping only links on this robot.
    points: dict[str, list] = {link_name: [] for link_name in link_names}
    if export_path.exists():
        with open(export_path, "r") as f:
            seed = json.load(f)
        for link_name, pts in seed.items():
            if link_name in points:
                points[link_name] = [list(map(float, p)) for p in pts]

    print(
        f"Loaded {urdf_path.name}: {robot.num_actuated_joints} DOF, {len(link_names)} links, "
        f"{len(local_meshes)} collision meshes, seeded {sum(len(v) for v in points.values())} points"
    )

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=0.5, height=0.5)

    initial_q = np.zeros(robot.num_actuated_joints, dtype=np.float32)

    with server.gui.add_folder("Contact Points"):
        place_mode = server.gui.add_checkbox("Place mode (click to add/remove)", initial_value=False)
        count_text = server.gui.add_text("Counts", initial_value="", disabled=True)
        undo_button = server.gui.add_button("Undo last")
        clear_link_dropdown = server.gui.add_dropdown("Link", tuple(link_names))
        only_selected_cb = server.gui.add_checkbox("Only show selected link", initial_value=False)
        clear_link_button = server.gui.add_button("Clear link")
        clear_all_button = server.gui.add_button("Clear all")
        export_button = server.gui.add_button("Export")

    with server.gui.add_folder("Joint Control", expand_by_default=False):
        joint_sliders = []
        for i, joint in enumerate(urdf.actuated_joints):
            lo = float(joint.limit.lower) if joint.limit and joint.limit.lower is not None else -np.pi
            hi = float(joint.limit.upper) if joint.limit and joint.limit.upper is not None else np.pi
            init = float(np.clip(initial_q[i], lo, hi))
            joint_sliders.append(server.gui.add_slider(joint.name, lo, hi, 1e-3, init))
        reset_button = server.gui.add_button("Reset Joints")

        @reset_button.on_click
        def _(_) -> None:
            for slider, init_val in zip(joint_sliders, initial_q):
                slider.value = float(init_val)

    # Mutable render state shared with the click handler.
    world_meshes: dict[str, trimesh.Trimesh] = {}
    link_poses = np.tile(np.eye(4, dtype=np.float32), (len(link_names), 1, 1))
    add_order: list[str] = []  # link_name per placed point, in click order, for Undo

    @undo_button.on_click
    def _(_) -> None:
        if add_order:
            points[add_order.pop()].pop()

    @clear_link_button.on_click
    def _(_) -> None:
        link = clear_link_dropdown.value
        points[link].clear()
        add_order[:] = [ln for ln in add_order if ln != link]

    @clear_all_button.on_click
    def _(_) -> None:
        for pts in points.values():
            pts.clear()
        add_order.clear()

    @export_button.on_click
    def _(_) -> None:
        out = {link: pts for link, pts in points.items() if pts}
        with open(export_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Exported {sum(len(v) for v in out.values())} points to {export_path}")

    def on_click(event: viser.ScenePointerEvent) -> None:
        if event.ray_origin is None or event.ray_direction is None:
            return
        ray_o = np.array(event.ray_origin, dtype=np.float64)
        ray_d = np.array(event.ray_direction, dtype=np.float64)
        best_link, best_point, best_dist = None, None, np.inf
        for link_name, mesh in world_meshes.items():
            locs = mesh.ray.intersects_location(ray_o[None], ray_d[None])[0]
            for loc in locs:
                dist = np.linalg.norm(loc - ray_o)
                if dist < best_dist:
                    best_link, best_point, best_dist = link_name, loc, dist
        if best_link is None or best_link != clear_link_dropdown.value:
            return  # Only edit the link you selected; a stray click on a neighbour is almost always a miss.
        # Remove if the hit lands near any existing point (world frame); else add it.
        for link_name in link_names:
            idx = link_names.index(link_name)
            for j, local_p in enumerate(points[link_name]):
                world_p = (link_poses[idx] @ np.array([*local_p, 1.0]))[:3]
                if np.linalg.norm(world_p - best_point) < args.remove_radius:
                    points[link_name].pop(j)
                    if link_name in add_order:
                        add_order.remove(link_name)
                    return
        idx = link_names.index(best_link)
        local = (np.linalg.inv(link_poses[idx]) @ np.array([*best_point, 1.0]))[:3]
        points[best_link].append([float(local[0]), float(local[1]), float(local[2])])
        add_order.append(best_link)

    # Register the click callback only in place mode: it captures clicks and disables orbit/drag.
    @place_mode.on_update
    def _(_) -> None:
        if place_mode.value:
            server.scene.on_pointer_event(event_type="click")(on_click)
        else:
            server.scene.remove_pointer_callback()

    mesh_handles: dict[str, viser.MeshHandle] = {}
    for link_name, mesh in local_meshes.items():
        mesh_handles[link_name] = server.scene.add_mesh_simple(
            f"/links/{link_name}", mesh.vertices, mesh.faces, color=ROBOT_COLOR, opacity=0.4, side="double"
        )

    def highlight_selected(_=None) -> None:
        for ln, handle in mesh_handles.items():
            selected = ln == clear_link_dropdown.value
            handle.color = (255, 220, 0) if selected else ROBOT_COLOR
            handle.visible = selected or not only_selected_cb.value

    clear_link_dropdown.on_update(highlight_selected)
    only_selected_cb.on_update(highlight_selected)
    for name, handle in mesh_handles.items():
        # Outside place mode a click on a link picks it, so you never hunt for a name in the dropdown.
        handle.on_click(lambda _, name=name: clear_link_dropdown.__setattr__("value", name))
    highlight_selected()

    contact_handle: Optional[viser.PointCloudHandle] = None
    last_q = None
    while True:
        q = np.array([s.value for s in joint_sliders], dtype=np.float32)
        if last_q is None or not np.array_equal(q, last_q):
            last_q = q
            state = robot.state(q=q)
            robot.forward_kinematics(state)
            # robokit returns each link pose as (position, wxyz); build the 4x4 the viser handles want.
            link_poses = np.zeros((len(link_names), 4, 4), dtype=np.float32)
            for i in range(len(link_names)):
                pose7 = state.get_T_world_link(i).numpy()[0]
                # Renormalize: float error in the quaternion makes trimesh reject the matrix as non-rigid.
                quat = pose7[3:] / np.linalg.norm(pose7[3:])
                link_poses[i] = trimesh.transformations.quaternion_matrix(quat)
                link_poses[i][:3, 3] = pose7[:3]
            for link_name, handle in mesh_handles.items():
                pose = link_poses[link_names.index(link_name)]
                handle.position = pose[:3, 3]
                handle.wxyz = trimesh.transformations.quaternion_from_matrix(pose)
                world_meshes[link_name] = local_meshes[link_name].copy().apply_transform(pose)

        world_pts = [
            (link_poses[link_names.index(ln)] @ np.array([*p, 1.0]))[:3] for ln in link_names for p in points[ln]
        ]
        if contact_handle is not None:
            contact_handle.remove()
            contact_handle = None
        if world_pts:
            world_pts = np.array(world_pts, dtype=np.float32)
            colors = np.tile(np.array([255, 60, 60], dtype=np.uint8), (len(world_pts), 1))
            contact_handle = server.scene.add_point_cloud(
                "/contact_points", world_pts, colors, point_size=args.point_size, point_shape="circle"
            )
        count_text.value = ", ".join(f"{ln}:{len(points[ln])}" for ln in link_names if points[ln]) or "(none)"

        time.sleep(max(1.0 / args.update_hz, 0.001))


if __name__ == "__main__":
    main(tyro.cli(Args))
