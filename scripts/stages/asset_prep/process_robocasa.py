"""
python scripts/stages/asset_prep/process_object_data.py
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh
import tyro
from rich.progress import track
from robokit.xform.numpy.rotation_conversions import quaternion_to_matrix

from graspkit.constants import DATA_ROOT


@dataclass
class Args:
    data_source: Literal["robocasa"] = "robocasa"
    """Data source to process."""

    robocasa_dir: str = str(DATA_ROOT / "object_data/robocasa")
    """Directory containing the RoboCasa dataset."""
    robocasa_processed_dir: str = str(DATA_ROOT / "object_data/robocasa/processed")
    """Directory to save the processed RoboCasa dataset."""


def process_robocasa(args: Args):
    objects_dir = os.path.join(args.robocasa_dir, "assets", "objects", "objaverse")
    for category in track(os.listdir(objects_dir), description="Processing categories"):
        if category == ".DS_Store":
            continue
        for obj_name in os.listdir(os.path.join(objects_dir, category)):
            if obj_name == ".DS_Store":
                continue
            try:
                xml_path = os.path.join(objects_dir, category, obj_name, "model.xml")
                base_dir = os.path.dirname(xml_path)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                visual_meshes = []
                collision_meshes = []

                for mesh_tag in root.findall(".//mesh"):
                    file = mesh_tag.attrib["file"]

                    scale_str = mesh_tag.attrib.get("scale", "1 1 1")
                    scale = list(map(float, scale_str.split()))

                    quat_str = mesh_tag.attrib.get("refquat", "1 0 0 0")
                    quat = list(map(float, quat_str.split()))

                    mesh = trimesh.load(os.path.join(base_dir, file))
                    mesh.apply_scale(scale)
                    tf = np.eye(4)
                    tf[:3, :3] = quaternion_to_matrix(np.array(quat))
                    mesh.apply_transform(tf)

                    if "visual" in file:
                        visual_meshes.append(mesh)
                    elif "collision" in file:
                        collision_meshes.append(mesh)
                    else:
                        raise ValueError(f"Unknown mesh type: {file}")

                if len(visual_meshes) > 0:
                    combined_visual = trimesh.Scene()
                    for mesh in visual_meshes:
                        combined_visual.add_geometry(mesh)
                    visual_mesh_path = os.path.join(args.robocasa_processed_dir, category, obj_name, "visual.glb")
                    if not os.path.exists(os.path.dirname(visual_mesh_path)):
                        os.makedirs(os.path.dirname(visual_mesh_path))
                    combined_visual.export(visual_mesh_path)

                if len(collision_meshes) > 0:
                    combined_collision = trimesh.util.concatenate(collision_meshes)
                    colision_mesh_path = os.path.join(args.robocasa_processed_dir, category, obj_name, "collision.glb")
                    if not os.path.exists(os.path.dirname(colision_mesh_path)):
                        os.makedirs(os.path.dirname(colision_mesh_path))
                    combined_collision.export(colision_mesh_path)

            except Exception as e:
                print(f"Error processing {obj_name}: {e}")


if __name__ == "__main__":
    args: Args = tyro.cli(Args, use_underscores=True)

    if args.data_source == "robocasa":
        process_robocasa(args)
    else:
        raise ValueError(f"Invalid data source: {args.data_source}")
