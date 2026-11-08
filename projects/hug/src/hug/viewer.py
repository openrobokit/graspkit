"""viser helpers shared by the sim and real entrypoints: server setup, browser tabs, pose plumbing, robot renders.

Importing this module never loads SAPIEN: :class:`RobotView`, :class:`GhostHands` and :class:`TrackingGhost` import
it when constructed, so the real scripts can use everything else with only the ``calibration`` extra.
"""

import atexit
import contextlib
import io
from typing import TYPE_CHECKING, Optional

import numpy as np
import viser
import yourdfpy
from numpy.typing import NDArray
from viser.extras import ViserUrdf

from hug import paths
from hug.sim.scene import (
    G1_JOINT_NAMES,
    HAND_MIMIC_FROM_URDF,
    HAND_MIMIC_SCALE,
    HAND_MIMIC_SUFFIXES,
    HAND_PROXIMAL_SUFFIXES,
)
from hug.utils import mat_to_pose7, pose7_to_mat
from hug.wbc.pose_estimator import G1_29DOF_HEAD_JOINT_NAMES, motor_q_29_to_32

if TYPE_CHECKING:
    import trimesh

    from hug.sim.backend import SapienBackend

# Per-stage ghost tint (RGBA): stages 1 and 2 share a pose and differ only in finger closure.
GHOST_STAGE_STYLE: list[tuple[str, tuple[float, float, float, float]]] = [
    ("pre-grasp", (0.25, 0.60, 1.00, 0.35)),
    ("grasp", (0.20, 0.90, 0.40, 0.45)),
    ("post-grasp", (1.00, 0.45, 0.15, 0.45)),
]


def make_server(port: int) -> viser.ViserServer:
    """A viser server with +z up, a ground grid and a key/fill lighting rig.

    Keep ambient low: with ``texture_material`` in place, a brighter ambient only washes the textures out."""
    with contextlib.redirect_stdout(io.StringIO()):  # viser's banner ignores verbose; the scripts print the URL
        server = viser.ViserServer(port=port, verbose=False)

    def stop_quietly() -> (
        None
    ):  # registered after viser's own exit hook, so it runs first and that one finds it stopped
        with contextlib.redirect_stdout(io.StringIO()):
            server.stop()

    atexit.register(stop_quietly)
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (1.5, 1.1, 1.35)  # open on the robot, front-left and a little above
    server.initial_camera.look_at = (0.05, 0.22, 0.6)
    server.scene.add_grid("/ground", width=3.0, height=3.0)
    server.scene.add_light_ambient("/light/ambient", intensity=0.4)
    server.scene.add_light_directional("/light/key", intensity=1.2, position=(1.5, 1.5, 3.0))
    server.scene.add_light_directional("/light/fill", intensity=0.4, position=(-1.5, -1.0, 2.0))
    return server


def texture_material(material) -> "trimesh.visual.material.PBRMaterial":
    """glTF material that shows an OBJ's texture as authored.

    trimesh would export ``Kd 0.4`` as a baseColorFactor and omit metallicFactor (glTF default 1.0): dim metal."""
    from trimesh.visual.material import PBRMaterial

    return PBRMaterial(baseColorTexture=material.image, metallicFactor=0.0, roughnessFactor=0.6)


def open_in_browser(*urls: str) -> None:
    """Open each URL in a browser tab; a no-op without a display, so an SSH session never launches a browser."""
    import os
    import sys
    import webbrowser

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform == "darwin"):
        return
    for url in urls:
        webbrowser.open_new_tab(url)


def set_pose(node, pose7: NDArray[np.float64]) -> None:
    """Place a viser node at a wxyz pose7."""
    node.position, node.wxyz = tuple(pose7[:3]), tuple(pose7[3:7])


def handle_pose7(handle: viser.TransformControlsHandle) -> NDArray[np.float64]:
    """viser handle → pose7 ``[x, y, z, qw, qx, qy, qz]``."""
    return np.concatenate([np.asarray(handle.position), np.asarray(handle.wxyz)]).astype(np.float64)


def world_T_W0_pose7(pelvis_pose7_W0: NDArray[np.float64], base_pose7: NDArray[np.float64]) -> NDArray[np.float64]:
    """World pose of the W0 frame: ``world_T_pelvis ∘ (W0_T_pelvis)⁻¹``.

    Control lives in W0, which travels with the robot; anchoring the render here makes a walking robot move."""
    return mat_to_pose7(pose7_to_mat(base_pose7) @ np.linalg.inv(pose7_to_mat(pelvis_pose7_W0)))


# The deployed robot's extras. Mount STL: mm, corner origin; the L515's back sits on its tilted plate's inner face,
# centred at MOUNT_SEAT_MM with outward normal MOUNT_SEAT_NORMAL, the plate's long edge along the camera's x axis.
MOUNT_STL = paths.REPO_ROOT / "hardware" / "l515_mount_60deg.stl"
MOUNT_SEAT_MM = np.array([40.1, 67.2, 30.9])
MOUNT_SEAT_NORMAL = np.array([0.0, -np.sin(np.pi / 3), -np.cos(np.pi / 3)])
L515_RADIUS_M, L515_DEPTH_M, L515_BACK_M = 0.0305, 0.026, 0.022  # body; optical origin to its back face
MOUNT_TO_CHEST_M = 0.012  # the calibrated pose leaves the plate this far off the model's chest: drawn seated on it
HEAD_ORIGIN_IN_TORSO = (0.0039635, 0.0, -0.044)  # stock head_link at logo_link, as Unitree's g1_29dof.urdf has it
ACTIVE_HEAD_LINKS = ("head_base_link", "camera_head_keypoint")  # the paper's head camera: not on the real robot


def real_robot_urdf(torso_T_cam: Optional[NDArray[np.float64]]) -> yourdfpy.URDF:
    """The G1 as deployed: the FK body with its stock head, the Inspire right hand, and the L515 on its chest mount.

    ``torso_T_cam=None`` leaves the mount and camera out. Actuated joints: the FK body's 32, then the hand's 6."""
    body = yourdfpy.URDF.load(str(paths.G1_29DOF_HEAD_URDF), load_meshes=False).robot
    hand = yourdfpy.URDF.load(str(paths.R_HAND_INSPIRE_URDF), load_meshes=False).robot
    for link in body.links + hand.links:  # the hand model marks each fingertip with a 1 cm cube
        if link.name in ACTIVE_HEAD_LINKS or link.name == "right_rubber_hand" or link.name.endswith("_tip"):
            link.visuals = []

    def part(name: str, geometry: yourdfpy.Geometry, rgba: tuple, torso_T: NDArray[np.float64]) -> None:
        color = yourdfpy.Material(name=name, color=yourdfpy.Color(rgba=np.array(rgba)))
        body.links.append(yourdfpy.Link(name, visuals=[yourdfpy.Visual(geometry=geometry, material=color)]))
        body.joints.append(yourdfpy.Joint(f"{name}_joint", "fixed", "torso_link", name, origin=torso_T))

    head_T = np.eye(4)
    head_T[:3, 3] = HEAD_ORIGIN_IN_TORSO
    head_mesh = paths.G1_29DOF_HEAD_URDF.parent.parent / "meshes" / "head_link.STL"
    part("stock_head", yourdfpy.Geometry(mesh=yourdfpy.Mesh(str(head_mesh))), (0.2, 0.2, 0.2, 1.0), head_T)
    if torso_T_cam is not None:
        # mount: rotate the seat normal onto the camera's optical axis and the STL x onto the camera's x, then seat it
        f, x = torso_T_cam[:3, 2], torso_T_cam[:3, 0]
        stl = np.stack([MOUNT_SEAT_NORMAL, [1.0, 0.0, 0.0], np.cross(MOUNT_SEAT_NORMAL, [1.0, 0.0, 0.0])], 1)
        mount_T = np.eye(4)
        mount_T[:3, :3] = np.stack([f, x, np.cross(f, x)], 1) @ np.linalg.inv(stl)
        # along the flat plate's normal, into the chest
        seat_back = MOUNT_TO_CHEST_M * (mount_T[:3, :3] @ [0.0, 0.0, 1.0])
        mount_T[:3, 3] = torso_T_cam[:3, 3] - L515_BACK_M * f - mount_T[:3, :3] @ (MOUNT_SEAT_MM * 1e-3) + seat_back
        mount = yourdfpy.Mesh(str(MOUNT_STL), scale=np.full(3, 1e-3))
        part("chest_mount", yourdfpy.Geometry(mesh=mount), (0.92, 0.92, 0.9, 1.0), mount_T)
        cam_T = torso_T_cam.copy()
        cam_T[:3, 3] += seat_back - (L515_BACK_M - L515_DEPTH_M / 2) * f  # the cylinder's axis is the optical axis
        l515 = yourdfpy.Cylinder(radius=L515_RADIUS_M, length=L515_DEPTH_M)
        part("l515", yourdfpy.Geometry(cylinder=l515), (0.15, 0.15, 0.15, 1.0), cam_T)

    # the body already mounts R_hand_base_link on the wrist, as the physics model does: swap in the articulated hand
    body.links = [link for link in body.links if link.name != "R_hand_base_link"] + hand.links
    body.joints += hand.joints
    return yourdfpy.URDF(robot=body, mesh_dir=str(paths.G1_29DOF_HEAD_URDF.parent))


class RealRobotView:
    """Render of the real G1 as deployed, from its 29 motor angles and the right hand's 6 measured joints.

    The stock head, the chest-mounted L515 at ``torso_T_cam`` and the Inspire right hand; placed at ``W0_T_pelvis``,
    the legs land on the floor the point clouds are lifted into."""

    def __init__(
        self, server: viser.ViserServer, torso_T_cam: Optional[NDArray[np.float64]] = None, root: str = "/robot"
    ) -> None:
        from hug.perception.object_cloud import load_torso_T_cam

        self.frame = server.scene.add_frame(root, show_axes=False)
        T = load_torso_T_cam(paths.TORSO_T_CAM) if torso_T_cam is None else torso_T_cam
        self._urdf = ViserUrdf(server, urdf_or_path=real_robot_urdf(T), root_node_name=root)
        names = list(self._urdf.get_actuated_joint_names())
        self._body = [names.index(n) for n in G1_29DOF_HEAD_JOINT_NAMES]
        self._hand = [i for i, n in enumerate(names) if n.startswith("R_")]  # thumb yaw, thumb pitch, index .. pinky
        self._cfg = np.zeros(len(names))

    def update(
        self,
        motor_q: np.ndarray,
        pelvis_pose7: Optional[NDArray[np.float64]] = None,
        right_hand: Optional[np.ndarray] = None,
    ) -> None:
        if pelvis_pose7 is not None:
            set_pose(self.frame, pelvis_pose7)
        self._cfg[self._body] = motor_q_29_to_32(motor_q)
        if right_hand is not None:
            self._cfg[self._hand] = right_hand
        self._urdf.update_cfg(self._cfg)


class RobotView:
    """Render of the simulated G1: body joints from ``motor_q``, hands and head read from the backend.

    Uses the backend's own mimic-free URDF, so the render can never disagree with the physics model."""

    def __init__(self, server: viser.ViserServer, backend: "SapienBackend", root: str = "/w0/robot") -> None:
        from hug.sim.backend import HEAD_JOINT_NAMES, patched_urdf

        self._backend = backend
        self.frame = server.scene.add_frame(root, show_axes=False)
        urdf = yourdfpy.URDF.load(str(patched_urdf(backend.urdf)))
        self._urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name=root)
        body_idx = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        self._src: list[tuple[str, int]] = []
        for name in urdf.actuated_joint_names:
            if name in body_idx:
                self._src.append(("body", body_idx[name]))
            elif name in HEAD_JOINT_NAMES:
                self._src.append(("head", HEAD_JOINT_NAMES.index(name)))
            elif name[2:] in HAND_PROXIMAL_SUFFIXES:
                self._src.append((f"{name[:1]}_prox", HAND_PROXIMAL_SUFFIXES.index(name[2:])))
            elif name[2:] in HAND_MIMIC_SUFFIXES:
                self._src.append((f"{name[:1]}_mimic", HAND_MIMIC_SUFFIXES.index(name[2:])))
            else:
                raise RuntimeError(f"unexpected joint {name!r} in {backend.urdf.name}")

    def update(self, motor_q: np.ndarray, root_pose7: Optional[NDArray[np.float64]] = None) -> None:
        if root_pose7 is not None:
            set_pose(self.frame, root_pose7)
        b = self._backend
        q = {
            "L_prox": b.read_hand("left"),
            "R_prox": b.read_hand("right"),
            "L_mimic": b.read_hand_mimic("left"),
            "R_mimic": b.read_hand_mimic("right"),
            "head": b.read_head(),
        }
        self._urdf.update_cfg(np.array([motor_q[i] if kind == "body" else q[kind][i] for kind, i in self._src]))


class GhostHands:
    """Translucent right Inspire hand at each predicted grasp stage (``R_hand_base_link`` poses in W0)."""

    def __init__(self, server: viser.ViserServer, parent: str = "/w0") -> None:
        from hug.sim.backend import patched_urdf

        self._server = server
        self._parent = parent
        self._nodes: list = []
        self.urdf = yourdfpy.URDF.load(str(patched_urdf(paths.R_HAND_INSPIRE_URDF)))
        names = list(self.urdf.actuated_joint_names)
        expected = {f"R_{s}" for s in (*HAND_PROXIMAL_SUFFIXES, *HAND_MIMIC_SUFFIXES)}
        if set(names) != expected:
            raise RuntimeError(f"{paths.R_HAND_INSPIRE_URDF.name} joints {names} != expected {sorted(expected)}")
        self._src = [
            ("prox", HAND_PROXIMAL_SUFFIXES.index(n[2:]))
            if n[2:] in HAND_PROXIMAL_SUFFIXES
            else ("mimic", HAND_MIMIC_SUFFIXES.index(n[2:]))
            for n in names
        ]
        self._mimic_lo = np.array([self.urdf.joint_map[f"R_{s}"].limit.lower for s in HAND_MIMIC_SUFFIXES])
        self._mimic_hi = np.array([self.urdf.joint_map[f"R_{s}"].limit.upper for s in HAND_MIMIC_SUFFIXES])

    def finger_cfg(self, hand_6: np.ndarray) -> np.ndarray:
        """Mimic joints posed the way the backend drives them: ``scale * proximal``, clamped."""
        q6 = np.asarray(hand_6, dtype=np.float64)
        mimic = np.clip(HAND_MIMIC_SCALE * q6[HAND_MIMIC_FROM_URDF], self._mimic_lo, self._mimic_hi)
        return np.array([q6[i] if kind == "prox" else mimic[i] for kind, i in self._src])

    def draw(self, stage_poses: list[np.ndarray], stage_hands: list[np.ndarray]) -> None:
        self.clear()
        for i, (name, rgba) in enumerate(GHOST_STAGE_STYLE):
            node = f"{self._parent}/ghost_{i}_{name}"
            frame = self._server.scene.add_frame(node, show_axes=False)
            set_pose(frame, stage_poses[i])
            ViserUrdf(self._server, urdf_or_path=self.urdf, root_node_name=node, mesh_color_override=rgba).update_cfg(
                self.finger_cfg(stage_hands[i])
            )
            self._nodes.append(frame)  # removing the frame drops the hand meshes under it

    def clear(self) -> None:
        for n in self._nodes:
            n.remove()
        self._nodes.clear()


class TrackingGhost:
    """Each hand's command: a translucent hand (right) or ball (left), with a red line to where the hand really is."""

    def __init__(self, server: viser.ViserServer, sides: tuple[str, ...], parent: str = "/w0") -> None:
        self._server, self._parent = server, parent
        self._frames = {
            s: server.scene.add_frame(f"{parent}/cmd_{s}", axes_length=0.05, axes_radius=0.003) for s in sides
        }
        self._hand: Optional[tuple[GhostHands, ViserUrdf]] = None
        self._line: Optional[viser.LineSegmentsHandle] = None
        if "right" in sides:
            ghost = GhostHands(server, parent)  # for its hand model and finger mapping
            self._hand = (
                ghost,
                ViserUrdf(
                    server, ghost.urdf, root_node_name=f"{parent}/cmd_right", mesh_color_override=(0.2, 0.9, 0.4, 0.3)
                ),
            )
        if "left" in sides:
            server.scene.add_icosphere(f"{parent}/cmd_left/ball", radius=0.02, color=(50, 230, 100), opacity=0.35)

    def update(self, cmd: dict[str, NDArray], hand: dict[str, NDArray], right_grip: Optional[NDArray] = None) -> None:
        """Pose the ghosts at ``cmd`` and draw the error lines to ``hand`` (pose7s in W0, keyed by side)."""
        for side, frame in self._frames.items():
            set_pose(frame, cmd[side])
        if self._hand is not None and right_grip is not None:
            self._hand[1].update_cfg(self._hand[0].finger_cfg(right_grip))
        points = np.array([[cmd[s][:3], hand[s][:3]] for s in self._frames], dtype=np.float32)
        self._line = self._server.scene.add_line_segments(
            f"{self._parent}/track_err", points, colors=(230, 40, 40), thickness=0.004
        )

    def set_visible(self, visible: bool) -> None:
        for node in (*self._frames.values(), self._line):
            if node is not None:
                node.visible = visible
