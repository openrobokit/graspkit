"""Shared fixtures; tests marked `cuda` or `hf_assets` skip without a CUDA device or a Hugging Face token."""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
import trimesh
import warp as wp
import yaml
from huggingface_hub import get_token
from robokit.geom.geoms import MeshGeom
from robokit.geom.scene import WarpScene

from graspkit.configs.robot_config import RobotConfig
from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    reasons = {
        "cuda": None if torch.cuda.is_available() else "needs a CUDA device",
        "hf_assets": None if get_token() else "needs Hugging Face assets (no HF token)",
    }
    for item in items:
        for marker, reason in reasons.items():
            if reason and item.get_closest_marker(marker):
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def box_scene() -> Callable[[float], WarpScene]:
    """Factory for a one-scene CPU `WarpScene` holding an origin-centered box with the given half extent."""

    def build(half_extent: float = 0.05) -> WarpScene:
        box = trimesh.creation.box(extents=(2 * half_extent,) * 3)
        mesh = wp.Mesh(
            points=wp.array(box.vertices, dtype=wp.vec3, device="cpu"),
            indices=wp.array(box.faces.ravel(), dtype=int, device="cpu"),
        )
        return WarpScene(1, "cpu").add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))

    return build


@pytest.fixture
def mano_config() -> RobotConfig:
    """Floating MANO right hand from robokit's Hugging Face assets; users need the `hf_assets` marker."""
    from robokit.assets.robots.end_effectors import mano_hand
    from robokit.utils.ee_coord_utils import HandCoordinateSpec

    return RobotConfig(
        name="mano_right",
        urdf_or_mjcf_path=str(mano_hand.URDF_PATH_RIGHT),
        mesh_dir=str(mano_hand.URDF_PATH_RIGHT.parent),
        collision_spheres_path=str(mano_hand.COLLISION_SPHERE_PATH_RIGHT),
        self_collision_ignore_path=str(mano_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
        base_type="floating",
        init_qpos=tuple([0.0] * 16),
        ee_coord_spec=HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z"),
        fingertip_link_names=["link_15", "link_3", "link_6", "link_12", "link_9"],
        thumb_link_names=["link_13", "link_14", "link_15"],
        middle_link_names=["link_4", "link_5", "link_6"],
        contact_sampling_qpos=(0.0,) * 13 + (0.65, 0.0, 0.0),
    )


@pytest.fixture
def cube_scene(tmp_path: Path) -> SceneDescription:
    """A 6 cm cube object at the origin above a table box."""
    cube_path = str(tmp_path / "cube.obj")
    trimesh.creation.box((0.06, 0.06, 0.06)).export(cube_path)
    return SceneDescription(
        scene_id="s0",
        target_actor_id="object",
        boxes={"table": BoxDescription("table", (0.5, 0.5, 0.05), (0.0, 0.0, -0.5))},
        meshes={"object": MeshDescription("object", cube_path, (1.0, 1.0, 1.0))},
    )


@pytest.fixture
def collision_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RobotConfig:
    """Three sibling links: only a/b overlap; sphere and capsule files ignore different pairs."""
    monkeypatch.setattr(logging.getLogger("graspkit"), "propagate", True)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="collision_test"><link name="base"/>'
        + "".join(
            f'<link name="{name}"><collision><geometry><box size="0.2 0.1 0.1"/></geometry></collision></link>'
            f'<joint name="{name}_joint" type="revolute"><parent link="base"/><child link="{name}"/>'
            f'<origin xyz="{x} 0 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            for name, x in (("a", 0), ("b", 0.15), ("c", 0.6))
        )
        + "</robot>"
    )
    spheres = tmp_path / "spheres.yml"
    spheres.write_text(
        yaml.safe_dump({"collision_spheres": {n: [{"center": [0, 0, 0], "radius": 0.2}] for n in "abc"}})
    )
    (tmp_path / "ignore.sphere.yml").write_text("self_collision_ignore: {a: [b]}\n")
    (tmp_path / "ignore.capsule.yml").write_text("self_collision_ignore: {a: [c]}\n")
    return RobotConfig(
        name="collision_test",
        urdf_or_mjcf_path=str(urdf),
        collision_spheres_path=str(spheres),
        self_collision_ignore_path=str(tmp_path / "ignore.sphere.yml"),
        palm_link_name="a",
        init_qpos=(0.0, 0.0, 0.0),
    )
