"""Visualize collision spheres and contact points of a fixed- or floating-base robot with viser.

Usage:
    uv run scripts/visualize/visualize_robot.py --robot-name xarm7_ability_right
    uv run scripts/visualize/visualize_robot.py --robot-name ability_right
"""

import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import trimesh
import tyro
import viser
import yourdfpy
from robokit.utils.visualize_utils import ViserMjcf
from robokit.xform.numpy import euler_angles_to_matrix, matrix_to_quaternion
from viser.extras import ViserUrdf

from graspkit.configs.robot import build_robot_config
from graspkit.constants import ROBOT_COLOR
from graspkit.robot.grasp_robot import GraspRobot, hand_coord_spec


@dataclass
class Args:
    robot_name: str = "mano_right"
    device: str = "cpu"
    update_hz: float = 30.0
    axis_length: float = 0.15
    """Length of the hand coord axis arrows (meters); bump up for larger robots."""


def create_joint_sliders(
    server: viser.ViserServer,
    joint_names: list[str],
    joint_limits: np.ndarray,
    initial_q: np.ndarray,
) -> list[viser.GuiInputHandle]:
    sliders: list[viser.GuiInputHandle] = []
    for joint_index, joint_name in enumerate(joint_names):
        lower, upper = joint_limits[joint_index]
        initial_value = float(initial_q[joint_index])
        lower = min(float(lower), initial_value)
        upper = max(float(upper), initial_value)
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_value,
        )
        sliders.append(slider)
    return sliders


def main(args: Args) -> None:
    robot_config = build_robot_config(args.robot_name)
    device = torch.device(args.device)
    robot_helper = GraspRobot.from_config(robot_config, device=device)
    is_floating = robot_helper.config.base_type == "floating"

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)

    root_frame = server.scene.add_frame("/robot", show_axes=False)
    # ViserMjcf (mujoco geoms) is the mjcf analog of ViserUrdf; both expose update_cfg(joint_vector).
    if robot_config.urdf_or_mjcf_path.endswith(".xml"):
        urdf_vis = ViserMjcf(server, robot_config.urdf_or_mjcf_path, root_node_name="/robot")
    else:
        urdf_load_kwargs = {
            "build_scene_graph": True,
            "build_collision_scene_graph": True,
            "load_meshes": True,
            "load_collision_meshes": True,
        }
        if robot_config.mesh_dir is not None:
            urdf_load_kwargs["mesh_dir"] = robot_config.mesh_dir
        urdf = yourdfpy.URDF.load(robot_config.urdf_or_mjcf_path, **urdf_load_kwargs)
        urdf_vis = ViserUrdf(
            server,
            urdf,
            root_node_name="/robot",
            mesh_color_override=ROBOT_COLOR,
            collision_mesh_color_override=ROBOT_COLOR,
        )

    joint_names = robot_helper.actuated_joint_names
    joint_limits = robot_helper.spec.actuated_joint_limits

    if is_floating:
        # For floating robots, joint_names and joint_limits exclude the 6 base joints.
        hand_joint_limits = joint_limits[: len(joint_names)]
        if robot_config.init_qpos is None:
            hand_initial_q = 0.5 * (hand_joint_limits[:, 0] + hand_joint_limits[:, 1])
            initial_q = np.concatenate([np.zeros(6, dtype=np.float32), hand_initial_q])
        else:
            initial_q = np.asarray(robot_config.init_qpos, dtype=np.float32)
            if initial_q.shape[0] == len(joint_names):
                initial_q = np.concatenate([np.zeros(6, dtype=np.float32), initial_q])
        if initial_q.shape[0] != len(joint_names) + 6:
            raise ValueError(
                f"init_qpos length ({initial_q.shape[0]}) does not match floating joint count "
                f"({len(joint_names) + 6}) or hand joint count ({len(joint_names)})"
            )
    else:
        if robot_config.init_qpos is None:
            initial_q = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
        else:
            initial_q = np.asarray(robot_config.init_qpos, dtype=np.float32)
        if initial_q.shape[0] != len(joint_names):
            raise ValueError(
                f"init_qpos length ({initial_q.shape[0]}) does not match actuated joint count ({len(joint_names)})"
            )

    has_avoid_points = robot_config.contact_avoid_points_path is not None
    with server.gui.add_folder("Settings"):
        show_spheres = server.gui.add_checkbox("Show Collision Spheres", initial_value=True)
        show_contacts = server.gui.add_checkbox("Show Contact Points", initial_value=False)
        show_hand_axes = server.gui.add_checkbox("Show Hand Coord Axes", initial_value=True)
        show_avoid = (
            server.gui.add_checkbox("Show Contact Avoid Points", initial_value=False) if has_avoid_points else None
        )

    avoid_link_indices: list[int] = []
    avoid_points_local: list[np.ndarray] = []
    if has_avoid_points:
        with open(robot_config.contact_avoid_points_path, "r") as avoid_file:
            raw_avoid_points = json.load(avoid_file)
        for link_name, points in raw_avoid_points.items():
            if len(points) == 0:
                continue
            avoid_link_indices.append(robot_helper.link_names.index(link_name))
            avoid_points_local.append(np.asarray(points, dtype=np.float32))

    base_sliders: dict[str, viser.GuiInputHandle] = {}
    if is_floating:
        base_initial_q = initial_q[:6]
        with server.gui.add_folder("Base Pose"):
            base_sliders["x"] = server.gui.add_slider(
                "x", min=-1.0, max=1.0, step=0.01, initial_value=float(base_initial_q[0])
            )
            base_sliders["y"] = server.gui.add_slider(
                "y", min=-1.0, max=1.0, step=0.01, initial_value=float(base_initial_q[1])
            )
            base_sliders["z"] = server.gui.add_slider(
                "z", min=0.0, max=2.0, step=0.01, initial_value=float(base_initial_q[2])
            )
            base_sliders["roll"] = server.gui.add_slider(
                "roll", min=-math.pi, max=math.pi, step=0.01, initial_value=float(base_initial_q[3])
            )
            base_sliders["pitch"] = server.gui.add_slider(
                "pitch", min=-math.pi, max=math.pi, step=0.01, initial_value=float(base_initial_q[4])
            )
            base_sliders["yaw"] = server.gui.add_slider(
                "yaw", min=-math.pi, max=math.pi, step=0.01, initial_value=float(base_initial_q[5])
            )
            reset_base_button = server.gui.add_button("Reset Base")

            @reset_base_button.on_click
            def _(_event: viser.GuiEvent) -> None:
                base_sliders["x"].value = float(base_initial_q[0])
                base_sliders["y"].value = float(base_initial_q[1])
                base_sliders["z"].value = float(base_initial_q[2])
                base_sliders["roll"].value = float(base_initial_q[3])
                base_sliders["pitch"].value = float(base_initial_q[4])
                base_sliders["yaw"].value = float(base_initial_q[5])

    with server.gui.add_folder("Joint Control"):
        if is_floating:
            # For floating robots, joint_names already excludes base joints
            hand_initial_q = initial_q[6:]
            joint_sliders = create_joint_sliders(server, joint_names, joint_limits[: len(joint_names)], hand_initial_q)
            reset_button = server.gui.add_button("Reset Joints")

            @reset_button.on_click
            def _(_event: viser.GuiEvent) -> None:
                for slider, init_val in zip(joint_sliders, hand_initial_q):
                    slider.value = float(init_val)
        else:
            joint_sliders = create_joint_sliders(server, joint_names, joint_limits, initial_q)
            reset_button = server.gui.add_button("Reset Joints")

            @reset_button.on_click
            def _(_event: viser.GuiEvent) -> None:
                for slider, init_val in zip(joint_sliders, initial_q):
                    slider.value = float(init_val)

    unit_sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere_handle: viser.BatchedMeshHandle | None = None
    contact_handle: viser.SceneNodeHandle | None = None
    avoid_handle: viser.SceneNodeHandle | None = None
    hand_axes_handles: list[viser.SceneNodeHandle] = []

    contact_indices = torch.arange(
        GraspRobot.get_contact_candidates(robot_helper.config, device)[0].shape[0], device=device
    ).unsqueeze(0)

    # Precompute hand coord axes info
    hand_base_link_idx = None
    if robot_config.ee_coord_spec:
        # Bare hands (e.g. mano_right) leave hand_base_link_name unset; the spec lives in the root link frame.
        hand_link_name = robot_config.hand_base_link_name or robot_helper.link_names[0]
        hand_base_link_idx = robot_helper.link_names.index(hand_link_name)
        # Semantic axes in the hand link frame, straight from the typed spec (migration target).
        spec = hand_coord_spec(robot_config)
        hand_front_dir = spec.palm_forward_vec  # palm forward
        hand_up_dir = spec.four_fingers_up_vec  # four fingers
        hand_right_dir = spec.palm_right_vec  # palm right

    # Set initial root frame position for floating base
    if is_floating:
        root_frame.position = tuple(initial_q[:3])
        root_frame.wxyz = tuple(matrix_to_quaternion(euler_angles_to_matrix(initial_q[3:6], "rxyz")))

    # ViserUrdf expects only hand joints for floating base (base is handled by root_frame)
    urdf_vis.update_cfg(initial_q[6:] if is_floating else initial_q)

    # Event-driven: rebuild the scene only when a slider/checkbox fires, so idle nodes don't churn.
    def update() -> None:
        nonlocal sphere_handle, contact_handle, avoid_handle
        hand_joint_q = np.array([slider.value for slider in joint_sliders], dtype=np.float32)

        if is_floating:
            x = base_sliders["x"].value
            y = base_sliders["y"].value
            z = base_sliders["z"].value
            roll = base_sliders["roll"].value
            pitch = base_sliders["pitch"].value
            yaw = base_sliders["yaw"].value

            root_frame.position = (x, y, z)
            root_frame.wxyz = matrix_to_quaternion(euler_angles_to_matrix(np.array([roll, pitch, yaw]), "rxyz"))

            # ViserUrdf handles floating base via root_frame, so only pass hand joints
            urdf_vis.update_cfg(hand_joint_q)

            # But robot_helper needs full configuration with base values
            base_qpos = np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)
            full_q = np.concatenate([base_qpos, hand_joint_q])
            full_q_tensor = torch.from_numpy(full_q).to(device).unsqueeze(0)
            link_poses = robot_helper.forward_kinematics_torch(full_q_tensor)
        else:
            urdf_vis.update_cfg(hand_joint_q)
            q_tensor = torch.from_numpy(hand_joint_q).to(device).unsqueeze(0)
            root_pose = torch.eye(4, device=device).unsqueeze(0)
            link_poses = robot_helper.forward_kinematics_torch(q_tensor, root_poses=root_pose)

        if show_spheres.value:
            collision_spheres = robot_helper.get_collision_spheres(link_poses)[0].cpu().numpy()
            positions = collision_spheres[:, :3]
            radii = collision_spheres[:, 3]
            wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (positions.shape[0], 1))
            if sphere_handle is not None:
                sphere_handle.remove()
            sphere_handle = server.scene.add_batched_meshes_simple(
                "/collision_spheres",
                vertices=unit_sphere_mesh.vertices,
                faces=unit_sphere_mesh.faces,
                batched_wxyzs=wxyzs,
                batched_positions=positions,
                batched_scales=radii,
                batched_colors=(0.0, 0.8, 0.2),
            )
        else:
            if sphere_handle is not None:
                sphere_handle.remove()
                sphere_handle = None

        if show_contacts.value:
            contact_points = robot_helper.get_contact_points(link_poses, contact_indices)[0].cpu().numpy()
            contact_colors = np.tile(np.array([255, 200, 0], dtype=np.uint8), (contact_points.shape[0], 1))
            if contact_handle is not None:
                contact_handle.remove()
            contact_handle = server.scene.add_point_cloud(
                "/contact_points",
                contact_points,
                contact_colors,
                point_size=0.004,
                point_shape="circle",
            )
        else:
            if contact_handle is not None:
                contact_handle.remove()
                contact_handle = None

        if show_avoid is not None and show_avoid.value:
            world_points = []
            for link_index, points_local in zip(avoid_link_indices, avoid_points_local):
                link_pose = link_poses[0, link_index].cpu().numpy()
                rot = link_pose[:3, :3]
                translation = link_pose[:3, 3]
                world_points.append(points_local @ rot.T + translation)
            avoid_world_points = np.concatenate(world_points, axis=0)
            avoid_colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (avoid_world_points.shape[0], 1))
            if avoid_handle is not None:
                avoid_handle.remove()
            avoid_handle = server.scene.add_point_cloud(
                "/contact_avoid_points",
                avoid_world_points,
                avoid_colors,
                point_size=0.006,
                point_shape="circle",
            )
        else:
            if avoid_handle is not None:
                avoid_handle.remove()
                avoid_handle = None

        if show_hand_axes.value and hand_base_link_idx is not None:
            for h in hand_axes_handles:
                h.remove()
            hand_axes_handles.clear()

            hand_pose = link_poses[0, hand_base_link_idx].cpu().numpy()
            origin = hand_pose[:3, 3]
            rot = hand_pose[:3, :3]
            axis_length = args.axis_length
            semantic_axes = [
                (hand_front_dir, "front", (255, 0, 0)),
                (hand_up_dir, "up (fingers)", (0, 255, 0)),
                (hand_right_dir, "right", (0, 0, 255)),
            ]
            for local_dir, label, color in semantic_axes:
                world_dir = rot @ local_dir
                end = origin + axis_length * world_dir
                h = server.scene.add_spline_catmull_rom(
                    f"/hand_axis_{label}",
                    points=np.stack([origin, end]),
                    thickness=10.0,
                    thickness_units="screen",
                    color=color,
                )
                hand_axes_handles.append(h)
                label_h = server.scene.add_label(
                    f"/hand_axis_label_{label}",
                    text=label,
                    wxyz=(1, 0, 0, 0),
                    position=tuple(end),
                )
                hand_axes_handles.append(label_h)
        else:
            for h in hand_axes_handles:
                h.remove()
            hand_axes_handles.clear()

    controls = [*joint_sliders, *base_sliders.values(), show_spheres, show_contacts, show_hand_axes]
    if show_avoid is not None:
        controls.append(show_avoid)
    for control in controls:
        control.on_update(lambda _: update())
    update()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Args))
