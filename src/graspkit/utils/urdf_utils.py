"""The simulation URDF of a `RobotConfig` and its hand joints, shared by every simulator backend."""

import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import yourdfpy
from robokit.assets import REVISION

from graspkit.configs.robot_config import RobotConfig
from graspkit.constants import DATA_ROOT

SIM_URDF_ROOT = DATA_ROOT / "sim_urdf" / REVISION
VIRTUAL_JOINTS = (("WRJTx", "prismatic", "1 0 0"), ("WRJTy", "prismatic", "0 1 0"), ("WRJTz", "prismatic", "0 0 1"),
                  ("WRJRx", "revolute", "1 0 0"), ("WRJRy", "revolute", "0 1 0"), ("WRJRz", "revolute", "0 0 1"))  # fmt: skip
"""The six base joints robokit's `*_floating_base` URDFs carry: xyz then intrinsic rxyz, matching the grasp euler."""


def export_sim_urdf(config: RobotConfig) -> tuple[Path, list[str]]:
    """Write `config`'s URDF minus `<mimic>`, plus virtual base joints if floating; return the path and hand joints."""
    floating = config.base_type == "floating"
    urdf = SIM_URDF_ROOT / f"{config.name}{'_floating_base' if floating else ''}.urdf"
    tree = ET.parse(config.urdf_or_mjcf_path)
    robot = tree.getroot()
    source = yourdfpy.URDF.load(config.urdf_or_mjcf_path, load_meshes=False)
    for mesh in robot.iter("mesh"):
        mesh.set("filename", str(Path(source._filename_handler(mesh.get("filename"))).resolve()))
    for joint in robot.findall("joint"):
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)
    if floating:
        links = [f"virtual_{name}_link" for name, _, _ in VIRTUAL_JOINTS] + [source.base_link]
        for i, (name, kind, axis) in enumerate(VIRTUAL_JOINTS):
            link = ET.SubElement(robot, "link", name=links[i])
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0.01")
            ET.SubElement(inertial, "inertia", ixx="0.0001", ixy="0", ixz="0", iyy="0.0001", iyz="0", izz="0.0001")
            joint = ET.SubElement(robot, "joint", name=name, type=kind)
            ET.SubElement(joint, "parent", link=links[i])
            ET.SubElement(joint, "child", link=links[i + 1])
            ET.SubElement(joint, "axis", xyz=axis)
            limit = 5.0 if kind == "prismatic" else 6.2832
            ET.SubElement(joint, "limit", lower=str(-limit), upper=str(limit), effort="10", velocity="3.14")
    parent = {joint.child: joint.parent for joint in source.robot.joints}
    hand_joints = []
    for joint in source.robot.joints:
        link = joint.child
        while link != config.hand_base_link_name and link in parent:
            link = parent[link]
        if joint.type != "fixed" and (floating or link == config.hand_base_link_name):
            hand_joints.append(joint.name)
    urdf.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=urdf.parent) as directory:
        generated = Path(directory) / urdf.name
        tree.write(generated)
        generated.replace(urdf)
    return urdf, hand_joints
