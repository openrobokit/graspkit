"""Per-source object mesh discovery: returns (names, collision_paths, visual_paths), visual None when unavailable."""

import glob
import os
from typing import Literal

from typing_extensions import deprecated


@deprecated("Use find_dexdiffuser_assets_v2 instead")
def find_dexdiffuser_assets_v1(data_dir: str) -> tuple[list[str], list[str], list[str] | None]:
    model_names, mesh_paths = [], []
    for source in ["egad", "kit", "contactdb", "ycb"]:
        for collision_path in glob.glob(os.path.join(data_dir, source, "*/*.stl")):
            name = os.path.basename(os.path.dirname(collision_path))
            model_names.append(f"{source}_{name}")
            mesh_paths.append(collision_path)
    return model_names, mesh_paths, None


def find_dexdiffuser_assets_v2(
    data_dir: str, sources: list[str] | None = None
) -> tuple[list[str], list[str], list[str] | None]:
    """Find dexdiffuser assets from the processed directory with collision.glb and visual.glb."""
    model_names, mesh_paths, visual_mesh_paths = [], [], []
    if sources is None:
        sources = ["egad", "kit", "contactdb", "ycb"]
    processed_dir = os.path.join(data_dir, "processed")
    for source in sources:
        source_dir = os.path.join(processed_dir, source)
        if not os.path.isdir(source_dir):
            continue
        for name in sorted(os.listdir(source_dir)):
            model_dir = os.path.join(source_dir, name)
            if not os.path.isdir(model_dir):
                continue
            mesh_path = os.path.join(model_dir, "collision.glb")
            visual_mesh_path = os.path.join(model_dir, "visual.glb")
            if os.path.exists(mesh_path) and os.path.exists(visual_mesh_path):
                model_names.append(f"{source}_{name}")
                mesh_paths.append(mesh_path)
                visual_mesh_paths.append(visual_mesh_path)
    return model_names, mesh_paths, visual_mesh_paths


def find_dexgraspnet_assets(
    data_dir: str, split_dir: str, split: Literal["train", "test"] = "test"
) -> tuple[list[str], list[str]]:
    """Enumerate dexgraspnet coacd/decomposed.obj paths for the given split (ycb/kit removed from train)."""
    with open(os.path.join(split_dir, "test.txt"), "r") as f:
        test_model_names = f.read().splitlines()
    if split == "train":
        model_names = [name for name in os.listdir(data_dir) if name not in test_model_names]
        model_names = [name for name in model_names if "ycb" not in name and "kit" not in name]
    else:
        model_names = test_model_names
    valid_model_names, mesh_paths = [], []
    for name in sorted(model_names):
        mesh_path = os.path.join(data_dir, name, "coacd/decomposed.obj")
        if os.path.exists(mesh_path):
            valid_model_names.append(name)
            mesh_paths.append(mesh_path)
    return valid_model_names, mesh_paths


def find_robocasa_assets(data_dir: str) -> tuple[list[str], list[str], list[str] | None]:
    categories = os.listdir(data_dir)
    robocasa_model_names, mesh_paths, visual_mesh_paths = [], [], []
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            model_names = os.listdir(category_dir)
            for model_name in model_names:
                if os.path.isdir(os.path.join(category_dir, model_name)):
                    mesh_path = os.path.join(category_dir, model_name, "collision.glb")
                    visual_mesh_path = os.path.join(category_dir, model_name, "visual.glb")
                    if os.path.exists(mesh_path) and os.path.exists(visual_mesh_path):
                        robocasa_model_names.append(model_name)
                        mesh_paths.append(mesh_path)
                        visual_mesh_paths.append(visual_mesh_path)
    return robocasa_model_names, mesh_paths, visual_mesh_paths


def find_objaverse_dex1b_assets(data_dir: str) -> tuple[list[str], list[str], list[str] | None]:
    mesh_paths = sorted(glob.glob(os.path.join(data_dir, "*.glb")))
    model_names = [os.path.splitext(os.path.basename(mesh_path))[0] for mesh_path in mesh_paths]
    return model_names, mesh_paths, None
