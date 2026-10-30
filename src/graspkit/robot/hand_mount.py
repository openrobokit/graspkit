"""Normalize hand frames; bake robokit's calibrated `hand_mount.json` adapter over the URDF's identity hand mount."""

import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yourdfpy
from robokit.utils.assembly_utils import compute_mount_transform
from robokit.utils.ee_coord_utils import MountCoordinateSpec
from robokit.xform.numpy import matrix_to_quaternion, pose7_to_tf_mat

from graspkit.constants import GRASPKIT_HOME

logger = logging.getLogger("graspkit")

MOUNT_CACHE_DIR = GRASPKIT_HOME / "cache" / "hand_mount"


def mjcf_with_hand_mount(
    mjcf_path: Path | str,
    hand_link: str | None = None,
    floating_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_mimics: tuple[tuple[str, str], ...] = (),
    floating_root: str | None = None,
) -> str:
    """Normalize a vendor hand root and commanded joint coupling without changing its dynamics."""
    source = Path(mjcf_path)
    key = str(source.resolve()).encode() + source.read_bytes()  # the patched file embeds absolute mesh paths
    key += json.dumps((hand_link, floating_offset, joint_mimics, floating_root)).encode()
    patched = MOUNT_CACHE_DIR / f"{source.stem}_{hashlib.sha256(key).hexdigest()[:12]}.xml"
    if not patched.exists():
        tree = ET.parse(source)
        root = tree.getroot()
        compiler = root.find("compiler")
        for directory in ("meshdir", "texturedir"):
            compiler.set(directory, str((source.parent / compiler.get(directory, "")).resolve()))
        if hand_link is not None:
            parent = next(e for e in root.iter() if e.find(f"./body[@name='{hand_link}']") is not None)
            hand = parent.find(f"./body[@name='{hand_link}']")
            T_parent_hand = pose7_to_tf_mat(
                np.r_[np.fromstring(hand.attrib["pos"], sep=" "), np.fromstring(hand.attrib["quat"], sep=" ")]
            )
            T_hand_root = compute_mount_transform(
                MountCoordinateSpec("z", "x", mount_point_xyz=tuple(-x for x in floating_offset)),
                MountCoordinateSpec("z", "x", mount_point_xyz=(0.0, 0.0, 0.0)),
            )
            T_parent_root = T_parent_hand @ T_hand_root
            frame = ET.Element(
                "body",
                name=f"{hand_link}_floating_root",
                pos=" ".join(map(str, T_parent_root[:3, 3])),
                quat=" ".join(map(str, matrix_to_quaternion(T_parent_root[:3, :3]))),
            )
            parent[list(parent).index(hand)] = frame
            hand.set("pos", " ".join(map(str, floating_offset)))
            hand.set("quat", "1 0 0 0")
            frame.append(hand)
        if joint_mimics:
            equality = ET.SubElement(root, "equality")
            for dependent, independent in joint_mimics:
                # Kinematic coupling metadata; original independent MuJoCo servos keep their dynamics.
                ET.SubElement(
                    equality,
                    "joint",
                    name=f"{dependent}_mimic",
                    joint1=dependent,
                    joint2=independent,
                    polycoef="0 1 0 0 0",
                    active="false",
                )
        MOUNT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tree.write(patched, encoding="utf-8", xml_declaration=True)
        if floating_root is not None:
            model = mujoco.MjSpec.from_file(str(patched))
            model.default.name = "floating_hand"
            floating = mujoco.MjSpec()
            floating.compiler.meshdir = model.compiler.meshdir
            floating.compiler.texturedir = model.compiler.texturedir
            hand = floating.worldbody.add_frame().attach_body(model.body(floating_root), "", "")
            root_joints = {joint.name for joint in hand.joints}
            for actuator in list(floating.actuators):
                if actuator.target in root_joints:
                    floating.delete(actuator)
            for joint in list(hand.joints):
                floating.delete(joint)
            hand.pos, hand.quat = (0, 0, 0), (1, 0, 0, 0)
            floating.compile()
            floating.to_file(str(patched))
    return str(patched)


def load_hand_mount(calib_path: Path | str) -> tuple[str, np.ndarray]:
    """Read a robokit ``hand_mount.json``, returning ``(joint_name, T_hand_mount [4, 4])``."""
    with open(calib_path) as f:
        calib = json.load(f)
    return calib["joint"], np.asarray(calib["T_hand_mount"], dtype=np.float64)


def urdf_with_hand_mount(urdf_path: Path | str, calib_path: Path | str | None) -> tuple[str, str]:
    """Bake the adapter into a URDF copy outside the asset cache, so re-fetches can't revert it; None is a no-op.

    Args:
        urdf_path: The nominal assembly URDF.
        calib_path: A ``hand_mount.json`` from robokit's ``handmount`` stage, or None.

    Returns:
        Tuple of ``(path to use as urdf_or_mjcf_path, directory to use as mesh_dir)``.

    """
    source = Path(urdf_path)
    if calib_path is None:
        return str(source), str(source.parent)

    joint_name, T_mount = load_hand_mount(calib_path)
    # Keyed by the source and the offset: a new asset revision or re-calibration lands on a new file, not a stale hit.
    digest = hashlib.sha256(str(source.resolve()).encode() + T_mount.tobytes() + joint_name.encode()).hexdigest()[:12]
    patched = MOUNT_CACHE_DIR / f"{source.stem}_{joint_name}_{digest}.urdf"
    if not patched.exists():
        urdf = yourdfpy.URDF.load(str(source), load_meshes=False, build_scene_graph=False)
        assert joint_name in urdf.joint_map, f"{joint_name} not in {source.name}"
        joint = urdf.joint_map[joint_name]
        assert joint.type == "fixed", f"{joint_name} is {joint.type}; the adapter must ride a fixed joint"
        joint.origin = np.asarray(joint.origin if joint.origin is not None else np.eye(4)) @ T_mount
        MOUNT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urdf.write_xml_file(str(patched))
        angle = np.degrees(np.arccos(np.clip((np.trace(T_mount[:3, :3]) - 1) / 2, -1, 1)))
        logger.info(
            f"hand mount: baked {np.linalg.norm(T_mount[:3, 3]) * 1000:.2f} mm / {angle:.2f} deg "
            f"into {joint_name} -> {patched}"
        )
    return str(patched), str(source.parent)
