import importlib.util
from pathlib import Path

import numpy as np
import sapien
import torch
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs import Pose
from robokit.xform.numpy import axis_angle_to_quaternion


def _get_maniskill_table_assets_dir() -> Path:
    spec = importlib.util.find_spec("mani_skill.utils.scene_builder.table")
    return Path(spec.submodule_search_locations[0]) / "assets"


DEFAULT_BOX_SIZE = (2.418, 1.209, 0.9196429)
DEFAULT_TABLE_SCALE = 1.75


class TableSceneBuilder(SceneBuilder):
    """TableSceneBuilder creates a table with customizable dimensions.

    The table with pose (0, 0, 0) (1, 0, 0, 0) is positioned with:
    - Longer side aligned with the x-axis
    - Bottom center aligned with the origin (0, 0, 0)
    - Height extending along the z-axis

    The table with pose (0, 0, -box_size[2]) (1, 0, 0, 0) is positioned with:
    - Longer side aligned with the x-axis
    - Top center aligned with the origin (0, 0, 0)
    - Height extending along the z-axis
    """

    def build(self, box_sizes: list[tuple[float, float, float]]):
        box_size = box_sizes[0]

        builder = self.scene.create_actor_builder()

        table_model_file = str(_get_maniskill_table_assets_dir() / "table.glb")

        table_pose = sapien.Pose(q=axis_angle_to_quaternion(np.array([0.0, 0.0, np.pi / 2])))
        box_pose = sapien.Pose(p=(0.0, 0.0, box_size[2] / 2))
        box_half_size = (box_size[0] / 2, box_size[1] / 2, box_size[2] / 2)
        builder.add_box_collision(pose=box_pose, half_size=box_half_size)

        visual_scale = [
            DEFAULT_TABLE_SCALE * box_size[1] / DEFAULT_BOX_SIZE[1],
            DEFAULT_TABLE_SCALE * box_size[0] / DEFAULT_BOX_SIZE[0],
            DEFAULT_TABLE_SCALE * box_size[2] / DEFAULT_BOX_SIZE[2],
        ]
        builder.add_visual_from_file(filename=table_model_file, scale=visual_scale, pose=table_pose)
        builder.initial_pose = sapien.Pose(p=(0.0, 0.0, -box_size[2]))
        self.table = builder.build_kinematic(name="table-workspace")

        self.scene_objects = [self.table]

    def initialize(self, env_idx: torch.Tensor, table_pose: Pose | sapien.Pose):
        # b = len(env_idx)
        self.table.set_pose(table_pose)
