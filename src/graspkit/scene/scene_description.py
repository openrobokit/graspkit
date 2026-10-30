"""Per-scene description with stable on-disk field names; adapted into graspkit ``SceneSpec``s via ``SceneGeom``."""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Literal, Optional, Tuple, Union

import msgspec
import numpy as np
import trimesh
from robokit.xform.numpy.rotation_conversions import quaternion_to_matrix

from graspkit.utils.io_utils import ensure_local_filepath


@dataclass
class BoxDescription:
    """Box primitive, usually for static elements like tables."""

    actor_id: str
    size: Tuple[float, float, float]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class MeshDescription:
    """Triangle mesh with actor id, optional visual-mesh path, and cached bounds."""

    actor_id: str
    collision_mesh_path: str
    scales: Tuple[float, float, float]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    visual_mesh_path: Optional[str] = None
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None

    def get_visual_mesh_path(self) -> str:
        """Return the visual mesh path, falling back to the collision mesh.

        >>> MeshDescription("o", "c.obj", (1.0, 1.0, 1.0)).get_visual_mesh_path()
        'c.obj'
        >>> MeshDescription("o", "c.obj", (1.0, 1.0, 1.0), visual_mesh_path="v.obj").get_visual_mesh_path()
        'v.obj'
        """
        return self.visual_mesh_path or self.collision_mesh_path

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        if self.bounds is not None:
            return self.bounds
        mesh = trimesh.load(self.collision_mesh_path, force="mesh", process=False)
        mesh.apply_scale(self.scales)
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = quaternion_to_matrix(np.asarray(self.quaternion))
        tf_mat[:3, 3] = self.position
        mesh.apply_transform(tf_mat)
        b = mesh.bounds.tolist()
        return ((b[0][0], b[1][0]), (b[0][1], b[1][1]), (b[0][2], b[1][2]))


@dataclass
class SceneDescription:
    """Per-scene geometry with a scene id and a manipulation-target actor id."""

    scene_id: str = ""
    target_actor_id: str = ""
    boxes: Dict[str, BoxDescription] = field(default_factory=dict)
    meshes: Dict[str, MeshDescription] = field(default_factory=dict)
    floating_scene_id: Optional[str] = None
    canonical_tabletop_scene_id: Optional[str] = None

    @staticmethod
    def load_scene_descriptions_msgpack(
        msgpack_path: str, return_splits: bool = False
    ) -> Union[Dict[str, "SceneDescription"], Dict[Literal["train", "test"], Dict[str, "SceneDescription"]]]:
        local_path = ensure_local_filepath(msgpack_path)
        with open(local_path, "rb") as f:
            data = msgspec.msgpack.decode(f.read(), type=Dict[Literal["train", "test"], Dict[str, SceneDescription]])
        return data if return_splits else {**data["train"], **data["test"]}

    @staticmethod
    def save_scene_descriptions_msgpack(
        scene_descriptions: Union[
            Dict[str, "SceneDescription"], Dict[Literal["train", "test"], Dict[str, "SceneDescription"]]
        ],
        msgpack_path: str,
    ) -> None:
        has_splits = isinstance(scene_descriptions, dict) and set(scene_descriptions.keys()) == {"train", "test"}
        payload = scene_descriptions if has_splits else {"train": scene_descriptions, "test": {}}
        with open(msgpack_path, "wb") as f:
            f.write(msgspec.msgpack.encode(payload))


@lru_cache(maxsize=None)
def get_table_only_scene_descriptions(
    table_size: Tuple[float, float, float] = (2.418, 1.209, 0.9196429),
    table_pos: Tuple[float, float, float] = (-0.12, 0.0, -0.9196429 / 2),
    table_quat: Tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678),
) -> SceneDescription:
    table_desc = BoxDescription(actor_id="table", size=table_size, position=table_pos, quaternion=table_quat)
    return SceneDescription(scene_id="table_only", boxes={"table": table_desc}, meshes={}, target_actor_id="table")


class SceneDescriptionsHelper:
    def __init__(self, scene_descriptions: List[SceneDescription]):
        self.scene_descriptions: List[SceneDescription] = scene_descriptions
        self._mesh_aux_info: Optional[Dict] = None
        self._actor_aux_info: Optional[Dict] = None

    @property
    def mesh_aux_info(self) -> Dict:
        if self._mesh_aux_info is None:
            mesh_key_to_indices = defaultdict(list)
            for scene_idx, scene_desc in enumerate(self.scene_descriptions):
                for mesh_actor_id, mesh_desc in scene_desc.meshes.items():
                    mesh_key_to_indices[(mesh_desc.collision_mesh_path, mesh_desc.scales)].append(
                        (scene_idx, mesh_actor_id)
                    )
            self._mesh_aux_info = {"mesh_key_to_indices": mesh_key_to_indices}
        return self._mesh_aux_info

    @property
    def actor_aux_info(self) -> Dict:
        if self._actor_aux_info is None:
            actor_id_to_pos, actor_id_to_quat = defaultdict(list), defaultdict(list)
            for scene_desc in self.scene_descriptions:
                for mesh_actor_id, mesh_desc in scene_desc.meshes.items():
                    actor_id_to_pos[mesh_actor_id].append(mesh_desc.position)
                    actor_id_to_quat[mesh_actor_id].append(mesh_desc.quaternion)
                for box_actor_id, box_desc in scene_desc.boxes.items():
                    actor_id_to_pos[box_actor_id].append(box_desc.position)
                    actor_id_to_quat[box_actor_id].append(box_desc.quaternion)
            self._actor_aux_info = {"actor_id_to_pos": actor_id_to_pos, "actor_id_to_quat": actor_id_to_quat}
        return self._actor_aux_info

    @property
    def mesh_keys(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        return list(self.mesh_aux_info["mesh_key_to_indices"].keys())

    @property
    def table_sizes(self) -> List[Tuple[float, float, float]]:
        return [tuple(scene_desc.boxes["table"].size) for scene_desc in self.scene_descriptions]

    def get_scene_indices_by_mesh_key(self, collision_path: str, mesh_scale: Tuple[float, float, float]) -> List[int]:
        return [idx_tuple[0] for idx_tuple in self.mesh_aux_info["mesh_key_to_indices"][collision_path, mesh_scale]]

    def get_actor_ids_by_mesh_key(self, collision_path: str, mesh_scale: Tuple[float, float, float]) -> List[str]:
        return [idx_tuple[1] for idx_tuple in self.mesh_aux_info["mesh_key_to_indices"][collision_path, mesh_scale]]

    def get_positions_by_actor_id(self, actor_id: str) -> List[Tuple[float, float, float]]:
        return self.actor_aux_info["actor_id_to_pos"][actor_id]

    def get_quaternions_by_actor_id(self, actor_id: str) -> List[Tuple[float, float, float, float]]:
        return self.actor_aux_info["actor_id_to_quat"][actor_id]
