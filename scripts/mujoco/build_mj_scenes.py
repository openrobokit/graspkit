"""Build the dexgraspnet-only scene subset (MjEnv lacks YCB .glb support) the MuJoCo experiments evaluate on.

uv run scripts/mujoco/build_mj_scenes.py --num-scenes 20 --tabletop
"""

import glob
import os

import tyro

from graspkit.assets.scene_descriptions import dex1b_v0_4
from graspkit.constants import DATA_ROOT, PROJECT_ROOT
from graspkit.scene.scene_description import SceneDescription


def main(num_scenes: int = 20, tabletop: bool = False) -> None:
    src = dex1b_v0_4.TABLETOP_MINI_PATH if tabletop else dex1b_v0_4.FLOATING_MINI_PATH
    name = "dexgraspnet_tabletop_subset" if tabletop else "dexgraspnet_subset"
    out_path = os.path.join(DATA_ROOT, f"scene_description/shadow_mj/{name}.msgpack")
    scenes = SceneDescription.load_scene_descriptions_msgpack(str(src))
    kept = {}
    for scene_id, scene in scenes.items():
        mesh = scene.meshes[scene.target_actor_id]
        parts_dir = os.path.join(PROJECT_ROOT, os.path.dirname(mesh.collision_mesh_path))
        if "dexgraspnet" not in mesh.collision_mesh_path:
            continue
        if not glob.glob(os.path.join(parts_dir, "*convex_piece_*.obj")):
            continue
        kept[scene_id] = scene
        if len(kept) >= num_scenes:
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    SceneDescription.save_scene_descriptions_msgpack(kept, out_path)
    print(f"wrote {len(kept)} dexgraspnet scenes -> {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
