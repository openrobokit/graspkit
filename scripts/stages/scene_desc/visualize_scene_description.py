"""
Visualize scene descriptions with an interactive viewer.

Usage:
uv run scripts/stages/scene_desc/visualize_scene_description.py data/scene_description/dex1b/v0.3/tabletop_mini.msgpack
"""

import logging

import numpy as np
import trimesh
import tyro
import viser
from robokit.xform.numpy import pose7_to_tf_mat
from tyro import conf
from viser import ViserServer

from graspkit.constants import OBJECT_COLOR, TABLE_COLOR
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.io_utils import get_cached_trimesh_from_file

logger = logging.getLogger("graspkit")


def visualize_scene_description(scene_description_path: conf.Positional[str]) -> None:
    """Visualize scene descriptions from a file with an interactive viewer.

    Args:
        scene_description_path: Path to the scene description file.
    """

    scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(scene_description_path, return_splits=False)
    logger.info(f"Loaded {len(scene_descriptions)} scene descriptions")
    scene_desc_candidates = list(scene_descriptions.values())

    viser_server = ViserServer()
    num_scenes = len(scene_desc_candidates)

    if num_scenes == 0:
        logger.warning("No scenes to visualize")
        return

    current_handles: list[viser.SceneNodeHandle] = []

    def load_scene(scene_index: int) -> None:
        """Load and display a scene, removing any previously loaded scene."""
        nonlocal current_handles

        for handle in current_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        current_handles.clear()

        scene_desc = scene_desc_candidates[scene_index]
        logger.info(f"Visualizing scene: {scene_desc.scene_id}")

        for actor_id, mesh_desc in scene_desc.meshes.items():
            mesh = get_cached_trimesh_from_file(mesh_desc.collision_mesh_path).copy()
            mesh.apply_scale(mesh_desc.scales)
            tf_mat = pose7_to_tf_mat(np.asarray([*mesh_desc.position, *mesh_desc.quaternion], dtype=np.float64))
            mesh.apply_transform(tf_mat)

            handle = viser_server.scene.add_mesh_simple(
                name=f"/scene/{actor_id}",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=OBJECT_COLOR if actor_id == scene_desc.target_actor_id else TABLE_COLOR,
                flat_shading=True,
            )
            current_handles.append(handle)

        for actor_id, box_desc in scene_desc.boxes.items():
            box_mesh = trimesh.creation.box(extents=box_desc.size)
            tf_mat = pose7_to_tf_mat(np.asarray([*box_desc.position, *box_desc.quaternion], dtype=np.float64))
            box_mesh.apply_transform(tf_mat)

            handle = viser_server.scene.add_mesh_simple(
                name=f"/scene/{actor_id}",
                vertices=box_mesh.vertices,
                faces=box_mesh.faces,
                color=TABLE_COLOR,
                flat_shading=True,
            )
            current_handles.append(handle)

    scene_slider = viser_server.gui.add_slider(
        "Scene Index", min=0, max=max(0, num_scenes - 1), step=1, initial_value=0
    )
    scene_slider.disabled = num_scenes <= 1

    scene_id_text = viser_server.gui.add_text("Scene ID", initial_value="")

    @scene_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        idx = int(scene_slider.value)
        load_scene(idx)
        scene_id_text.value = scene_desc_candidates[idx].scene_id

    # Initial load
    load_scene(0)
    scene_id_text.value = scene_desc_candidates[0].scene_id

    logger.info(f"Visualizing {num_scenes} scenes. Use the slider to navigate.")
    import time

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(visualize_scene_description, use_underscores=True)
