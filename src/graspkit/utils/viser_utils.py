"""Viser helpers: scene drawing and an interactive viewer for recorded rollouts."""

import glob
import os
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import viser
import yourdfpy
from robokit.xform.numpy import euler_angles_to_matrix, matrix_to_quaternion
from viser.extras import ViserUrdf

from graspkit.constants import GRASPKIT_HOME, OBJECT_COLOR, ROBOT_COLOR, TABLE_COLOR
from graspkit.datasets.replay_data import ReplayData
from graspkit.scene.scene_description import SceneDescription

if TYPE_CHECKING:
    from robokit.utils.visualize_utils import ViserMjcf


def visualize_scene_meshes(server: viser.ViserServer, scene_desc: SceneDescription) -> None:
    """Draw one scene's meshes/boxes, removing the previously drawn scene first."""
    scene_prefix = "/current_scene"
    server.scene.remove_by_name(scene_prefix)

    for mesh_name, mesh_desc in scene_desc.meshes.items():
        mesh_path = os.path.join(str(GRASPKIT_HOME), mesh_desc.collision_mesh_path)
        mesh = trimesh.load(mesh_path, force="mesh", process=False, skip_materials=True)
        scale = mesh_desc.scales[0] if np.allclose(mesh_desc.scales, mesh_desc.scales[0]) else 1.0

        server.scene.add_mesh_simple(
            f"{scene_prefix}/mesh_{mesh_name}",
            vertices=mesh.vertices * scale,
            faces=mesh.faces,
            color=OBJECT_COLOR if mesh_name == scene_desc.target_actor_id else TABLE_COLOR,
            flat_shading=True,
            position=np.array(mesh_desc.position),
            wxyz=np.array(mesh_desc.quaternion),
        )

    for box_name, box_desc in scene_desc.boxes.items():
        box_mesh = trimesh.creation.box(box_desc.size)
        server.scene.add_mesh_simple(
            f"{scene_prefix}/box_{box_name}",
            vertices=box_mesh.vertices,
            faces=box_mesh.faces,
            color=TABLE_COLOR,
            flat_shading=True,
            position=np.array(box_desc.position),
            wxyz=np.array(box_desc.quaternion),
        )


class ReplayViewer:
    def __init__(self, port: int = 8080):
        self.replays: list[ReplayData] = []
        self.shown: list[ReplayData] = []
        self.current: ReplayData | None = None
        self.robot: "ViserUrdf | ViserMjcf | None" = None
        self.robot_path = ""
        self.object: viser.MeshHandle | None = None
        self.object_collision: viser.MeshHandle | None = None
        self.lock = threading.RLock()
        self.server = viser.ViserServer(port=port)
        self.root = self.server.scene.add_frame("/robot", show_axes=False)
        self.info = self.server.gui.add_markdown("**no rollouts**")
        self.stage_filter = self.server.gui.add_dropdown("stage", options=["-"], initial_value="-")
        self.label_filter = self.server.gui.add_dropdown("label", options=["all"], initial_value="all")
        self.rollout_slider = self.server.gui.add_slider("rollout", min=0, max=1, step=1, initial_value=0)
        self.frame_slider = self.server.gui.add_slider("frame", min=0, max=1, step=1, initial_value=0)
        self.fps_slider = self.server.gui.add_slider("fps", min=1, max=240, step=1, initial_value=60)
        self.play_checkbox = self.server.gui.add_checkbox("Play", initial_value=False)
        self.mesh_type = self.server.gui.add_dropdown(
            "mesh", options=["visual", "collision", "both"], initial_value="visual"
        )
        self.mesh_type.on_update(lambda _: self._update_mesh())
        self.stage_filter.on_update(lambda _: self._apply_filters())
        self.label_filter.on_update(lambda _: self._apply_filters())
        self.rollout_slider.on_update(lambda _: self._show_replay(int(self.rollout_slider.value)))
        self.frame_slider.on_update(lambda _: self._show_frame(int(self.frame_slider.value)))
        threading.Thread(target=self._animate, daemon=True).start()

    def _animate(self) -> None:
        while True:
            if self.play_checkbox.value and self.current is not None:
                self.frame_slider.value = (int(self.frame_slider.value) + 1) % (int(self.frame_slider.max) + 1)
            time.sleep(1.0 / self.fps_slider.value)

    def set_replays(self, replays: list[ReplayData]) -> None:
        with self.lock:  # the gallery swaps replay sets from a worker thread while frames animate
            self.replays = replays
            replay = replays[0]
            if replay.robot_path != self.robot_path:
                if self.robot is not None:
                    self.server.scene.remove_by_name("/robot")  # ViserUrdf.remove() warns re-removing child nodes
                self.robot_path = replay.robot_path
                if replay.robot_path.endswith(".xml"):
                    from robokit.utils.visualize_utils import ViserMjcf

                    self.robot = ViserMjcf(self.server, replay.robot_path, root_node_name="/robot")
                else:
                    kwargs = {"build_scene_graph": True, "load_meshes": True}
                    kwargs |= {"build_collision_scene_graph": True, "load_collision_meshes": True}
                    if replay.mesh_dir:
                        kwargs["mesh_dir"] = replay.mesh_dir
                    self.robot = ViserUrdf(
                        self.server,
                        yourdfpy.URDF.load(replay.robot_path, **kwargs),
                        root_node_name="/robot",
                        mesh_color_override=ROBOT_COLOR,
                        collision_mesh_color_override=ROBOT_COLOR,
                        load_collision_meshes=True,
                    )
                self._update_mesh()
            self.stage_filter.options = list(dict.fromkeys(r.stage for r in replays))  # pipeline order
            self.label_filter.options = ["all"] + sorted({r.label for r in replays})
            self.stage_filter.value = self.stage_filter.options[0]
            self._apply_filters()

    def _update_mesh(self) -> None:
        self.robot.show_visual = self.mesh_type.value != "collision"
        self.robot.show_collision = self.mesh_type.value != "visual"
        # The object collision variant only exists for convex-piece folders; keep visual as fallback.
        if self.object is not None:
            self.object.visible = self.mesh_type.value != "collision" or self.object_collision is None
        if self.object_collision is not None:
            self.object_collision.visible = self.mesh_type.value != "visual"

    def _apply_filters(self) -> None:
        stage, label = self.stage_filter.value, self.label_filter.value
        self.shown = [r for r in self.replays if r.stage == stage and (label == "all" or r.label == label)]
        self.rollout_slider.max = max(1, len(self.shown) - 1)
        self.frame_slider.max = max([len(r.qpos) for r in self.shown] + [1]) - 1
        self.rollout_slider.value = min(int(self.rollout_slider.value), self.rollout_slider.max)
        self._show_replay(int(self.rollout_slider.value))

    def _show_replay(self, index: int) -> None:
        with self.lock:
            if not self.shown:
                self.current = None
                self.info.content = "**no rollouts match**"
                return
            replay = self.shown[min(index, len(self.shown) - 1)]  # slider may lag a filter change
            self.current = replay
            if self.object is not None:
                self.server.scene.remove_by_name("/object")
            mesh = trimesh.load(replay.obj_mesh_path, force="mesh", process=False)
            self.object = self.server.scene.add_mesh_simple(
                "/object",
                vertices=np.asarray(mesh.vertices) * replay.obj_scale,
                faces=np.asarray(mesh.faces),
                color=OBJECT_COLOR,
                flat_shading=False,
            )
            if self.object_collision is not None:
                self.server.scene.remove_by_name("/object_collision")
            self.object_collision = None
            pieces = sorted(glob.glob(os.path.join(os.path.dirname(replay.obj_mesh_path), "*convex_piece_*.obj")))
            if pieces:
                coll = trimesh.util.concatenate([trimesh.load(p, force="mesh", process=False) for p in pieces])
                self.object_collision = self.server.scene.add_mesh_simple(
                    "/object_collision",
                    vertices=np.asarray(coll.vertices) * replay.obj_scale,
                    faces=np.asarray(coll.faces),
                    color=ROBOT_COLOR,
                    wireframe=True,
                )
            self._update_mesh()
            self.server.scene.remove_by_name("/table")
            if float(np.max(replay.table_size)) > 0:
                box = trimesh.creation.box(replay.table_size)
                self.server.scene.add_mesh_simple(
                    "/table",
                    vertices=np.asarray(box.vertices),
                    faces=np.asarray(box.faces),
                    color=TABLE_COLOR,
                    flat_shading=True,
                    position=tuple(replay.table_pos),
                    wxyz=tuple(replay.table_quat),
                )
            self.info.content = f"**{index}/{len(self.shown) - 1}** — {replay.stage} / {replay.label}"
            self._show_frame(int(self.frame_slider.value))

    def _show_frame(self, frame: int) -> None:
        with self.lock:
            replay = self.current
            if replay is None:
                return
            frame = min(frame, len(replay.qpos) - 1)
            qpos = replay.qpos[frame]
            if replay.base_type == "floating":
                self.root.position = tuple(qpos[:3])
                self.root.wxyz = tuple(matrix_to_quaternion(euler_angles_to_matrix(qpos[3:6], "rxyz")))
                self.robot.update_cfg(np.asarray(qpos[6:]))
            else:
                self.root.position = tuple(replay.root_pos)
                self.root.wxyz = tuple(replay.root_quat)
                self.robot.update_cfg(np.asarray(qpos))
            self.object.position = tuple(replay.obj_pose[frame, :3])
            self.object.wxyz = tuple(replay.obj_pose[frame, 3:7])
            if self.object_collision is not None:
                self.object_collision.position = tuple(replay.obj_pose[frame, :3])
                self.object_collision.wxyz = tuple(replay.obj_pose[frame, 3:7])
