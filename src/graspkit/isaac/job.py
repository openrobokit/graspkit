"""Export the native robot (joints, cameras, contact links, materials) and tabletop scenes for the Isaac worker."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import trimesh
import yourdfpy

from graspkit.configs.env_config import CameraSpec
from graspkit.configs.robot_config import RobotConfig
from graspkit.constants import GRASPKIT_HOME, LOG_ROOT
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.urdf_utils import export_sim_urdf


def export_robot(config: RobotConfig, cameras: tuple[CameraSpec, ...]) -> dict:
    """Isaac/MuJoCo robot export: sim URDF, chain-qpos-to-joint affine map (mimics, parked branches), cameras."""
    assert config.base_type == "fixed", "Isaac and MuJoCo support fixed-base URDF robots."
    rig = GraspRobot.from_config(config, device="cpu")
    full = GraspRobot.from_config(config, device="cpu", prune=False)
    sim_urdf, hand = export_sim_urdf(config)
    source = yourdfpy.URDF.load(config.urdf_or_mjcf_path, load_meshes=False)
    joints = {joint.name: joint for joint in source.robot.joints if joint.type != "fixed"}
    affine = {}
    for name, joint in joints.items():
        if joint.mimic is None:
            coefficient = np.zeros(rig.num_actuated_joints, np.float32)
            if name in rig.actuated_joint_names:
                coefficient[rig.actuated_joint_names.index(name)] = 1
                bias = 0.0
            else:
                bias = float(full.init_qpos[full.actuated_joint_names.index(name)])
            affine[name] = (coefficient, bias)
    pending = {name: joint for name, joint in joints.items() if joint.mimic is not None}
    while pending:
        ready = [name for name, joint in pending.items() if joint.mimic.joint in affine]
        assert ready, "Cyclic mimic graph."
        for name in ready:
            mimic = pending.pop(name).mimic
            coefficient, bias = affine[mimic.joint]
            affine[name] = (coefficient * mimic.multiplier, bias * mimic.multiplier + mimic.offset)
    rows = []
    for camera in cameras:
        intrinsic = camera.intrinsic
        if intrinsic is None:
            focal = camera.height / (2 * np.tan(np.deg2rad(camera.fov) / 2))
            intrinsic = ((focal, 0, camera.width / 2), (0, focal, camera.height / 2), (0, 0, 1))
        rows.append(
            {
                "mount": camera.mount,
                "pose": list(camera.pose),
                "intrinsic": np.asarray(intrinsic).tolist(),
                "width": camera.width,
                "height": camera.height,
            }
        )
    # Isaac's importer needs mass on every link.
    tree = ET.parse(sim_urdf)
    for link in tree.getroot().iter("link"):
        if link.find("inertial") is None and link.find("collision") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value="0.000001")
            ET.SubElement(
                inertial, "inertia", ixx="0.000001", ixy="0", ixz="0", iyy="0.000001", iyz="0", izz="0.000001"
            )
    return {
        "tree": tree,
        "coefficients": np.stack([affine[name][0] for name in joints]),
        "bias": np.asarray([affine[name][1] for name in joints], np.float32),
        "init_qpos": rig.init_qpos.numpy(),
        "root_pose": rig.root_pose_7d.numpy(),
        "joint_names": list(joints),
        "policy_joint_names": rig.actuated_joint_names,
        "hand_joints": hand,
        "contact_links": list(config.fingertip_link_names) + ([config.palm_link_name] if config.palm_link_name else []),
        "tip_links": list(config.fingertip_link_names),
        "cameras": rows,
    }


def prepare_robot(config: RobotConfig, cameras: tuple[CameraSpec, ...], directory: Path, appearance: Path) -> dict:
    """`export_robot` with OBJ meshes for Isaac's importer, plus SAPIEN's native appearance from `appearance`."""
    robot = export_robot(config, cameras)
    tree = robot["tree"]
    # Isaac reads OBJ, not GLB; converted meshes are shared across runs.
    for element in tree.getroot().iter("mesh"):
        original = Path(element.attrib["filename"])
        if original.suffix.lower() == ".glb":
            converted = LOG_ROOT / "isaac/meshes" / hashlib.sha256(original.read_bytes()).hexdigest() / "mesh.obj"
            if not converted.exists():
                converted.parent.mkdir(parents=True, exist_ok=True)
                mesh = trimesh.load(original, process=False)
                for geometry in mesh.geometry.values():
                    if geometry.visual.kind == "texture" and geometry.visual.uv is None:
                        geometry.visual.uv = np.zeros((len(geometry.vertices), 2))
                mesh.export(converted, file_type="obj", digits=12, include_normals=True)
            element.set("filename", str(converted))
    urdf = directory / "robot.urdf"
    tree.write(urdf)
    contract = json.loads((appearance / "native_robot_materials/contract.json").read_text())
    key, native = next(
        (key, entry) for key, entry in contract["robots"].items() if entry["robot_config_name"] == config.name
    )
    camera_materials = {}
    if key == "agilex_aloha":  # its camera GLB exports as several OBJ material groups, each matched to SAPIEN's
        mapping = json.loads((appearance / "native_robot_materials/aloha_camera_material_mapping.json").read_text())
        assert mapping["all16_groups_mapped_exactly_once"]
        for part in mapping["parts"]:
            for group in part["source_OBJ_groups"]:
                camera_materials.setdefault(part["link"], {})[group["OBJ_group"]] = part[
                    "native_effective_material"
                ] | {"expected_faces": group["faces"]}
    table_visual = appearance / "native_appearance/table_visual.h5"
    textures = {}
    with h5py.File(table_visual) as table:
        for index in table["textures"]:
            textures[index] = str(directory / f"table_texture_{index}.png")
            Path(textures[index]).write_bytes(table[f"textures/{index}/png"][...].tobytes())
    names = ("joint_names", "policy_joint_names", "hand_joints", "contact_links", "tip_links")
    return {
        **{key: robot[key] for key in ("coefficients", "bias", "init_qpos", "root_pose")},
        "attrs": {
            "urdf": str(urdf),
            **{key: robot[key] for key in names},
            "cameras": json.dumps(robot["cameras"]),
            "materials": json.dumps(native["effective_native_material_getters"]),
            "camera_materials": json.dumps(camera_materials),
            "table_visual": str(table_visual),
            "table_textures": json.dumps(textures),
        },
    }


def write_job(path: Path, scenes: list[SceneDescription], qpos: np.ndarray, root_pose: np.ndarray, attrs: dict):
    """One env per scene: scaled convex pieces and visuals, table/object poses, full-joint `qpos` targets (N, T, J)."""
    assert qpos.shape[0] == len(scenes) and qpos.ndim == 3
    objects = [scene.meshes[scene.target_actor_id] for scene in scenes]
    keys = sorted({(mesh.collision_mesh_path, tuple(mesh.scales)) for mesh in objects})
    with h5py.File(path, "w") as job:
        for index, (mesh_path, scale) in enumerate(keys):
            mesh = trimesh.load(GRASPKIT_HOME / mesh_path, process=False, force="mesh")
            scale = np.asarray(scale, np.float32)
            for piece, part in enumerate(mesh.split(only_watertight=False)):
                job[f"meshes/{index}/{piece}/vertices"] = np.asarray(part.vertices, np.float32) * scale
                job[f"meshes/{index}/{piece}/faces"] = np.asarray(part.faces, np.int32)
            job[f"visuals/{index}/vertices"] = np.asarray(mesh.vertices, np.float32) * scale
            job[f"visuals/{index}/faces"] = np.asarray(mesh.faces, np.int32)
            colors = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
            job[f"visuals/{index}/colors"] = (
                np.full((len(mesh.vertices), 3), 0.6, np.float32)  # SAPIEN's default for untextured meshes
                if mesh.visual.kind is None
                else np.asarray(colors.vertex_colors[:, :3], np.float32) / 255
            )
        tables = [scene.boxes["table"] for scene in scenes]
        job["scene_id"] = np.asarray([scene.scene_id for scene in scenes], dtype="S")
        job["mesh_idx"] = [keys.index((mesh.collision_mesh_path, tuple(mesh.scales))) for mesh in objects]
        job["obj_pose"] = np.asarray([[*mesh.position, *mesh.quaternion] for mesh in objects], np.float32)
        job["table_pose"] = np.asarray([[*table.position, *table.quaternion] for table in tables], np.float32)
        job["table_size"] = np.asarray([table.size for table in tables], np.float32)
        job["root_pose"] = np.broadcast_to(root_pose, (len(scenes), 7)).astype(np.float32)
        job["qpos"] = qpos.astype(np.float32)
        job.attrs.update(attrs)
