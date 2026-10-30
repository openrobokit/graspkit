"""MJCF->SAPIEN loader, adapted from NVIDIA warp's import_mjcf.py; a Mujoco articulation is a <body> tag tree.

- <geom> tags in <worldbody> become separate static actors; unloadable data warns only if verbosity is on (default off).
- Joint solver/stiffness/actuator properties aren't imported; wire them up via a SAPIEN controller like other robots.
- Contact tags aren't supported; tendons/equality constraints are supported but may not behave identically.
- Geom render groups 0/2 are visible by default; other groups get no visual body (SAPIEN can't toggle render groups).
- contype=0 means no collisions, so we skip the collision shape; worldbody geoms outside <body> are treated as static.
"""

import math
import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, no_type_check
from xml.etree.ElementTree import Element

import numpy as np
import sapien
from robokit.xform.numpy import axis_angle_to_quaternion, euler_angles_to_matrix, matrix_to_quaternion
from sapien import ActorBuilder, Pose
from sapien.physx import PhysxArticulation, PhysxMaterial
from sapien.render import RenderMaterial, RenderTexture2D
from sapien.wrapper.articulation_builder import (
    ArticulationBuilder,
    LinkBuilder,
    MimicJointRecord,
)


@no_type_check
@dataclass
class MJCFTexture:
    name: str
    type: Literal["skybox", "cube", "2d"]
    rgb1: list
    rgb2: list
    file: str


# NOTE: contact=False to disable self-collision
DEFAULT_MJCF_OPTIONS = dict(contact=False)


def _parse_int(attrib, key, default):
    if key in attrib:
        return int(attrib[key])
    else:
        return default


def _parse_float(attrib, key, default):
    if key in attrib:
        return float(attrib[key])
    else:
        return default


def _str_to_float(string: str, delimiter=" "):
    res = [float(x) for x in string.split(delimiter)]
    if len(res) == 1:
        return res[0]
    return res


def _merge_attrib(default_attrib: dict, incoming_attribs: Union[List[dict], dict]):
    def helper_merge(a: dict, b: dict, path=[]):
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    helper_merge(a[key], b[key], path + [str(key)])
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a

    attrib = deepcopy(default_attrib)
    if isinstance(incoming_attribs, dict):
        incoming_attribs = [incoming_attribs]
    reduce(helper_merge, [attrib] + incoming_attribs)
    return attrib


def _parse_vec(attrib, key, default):
    if key in attrib:
        out = np.fromstring(attrib[key], sep=" ", dtype=np.float32)
    else:
        out = np.array(default, dtype=np.float32)
    return out


def _parse_orientation(attrib, use_degrees, euler_seq):
    if "quat" in attrib:
        wxyz = np.fromstring(attrib["quat"], sep=" ")
        return wxyz
    if "euler" in attrib:
        euler_angles = np.fromstring(attrib["euler"], sep=" ")
        if use_degrees:
            euler_angles *= np.pi / 180
        # TODO (stao): support other axes?
        return matrix_to_quaternion(euler_angles_to_matrix(euler_angles * np.array([1.0, -1.0, 1.0])))
    if "axisangle" in attrib:
        axisangle = np.fromstring(attrib["axisangle"], sep=" ")
        angle = axisangle[3]
        if use_degrees:
            angle *= np.pi / 180
        axis = axisangle[:3] / np.linalg.norm(axisangle[:3])
        return axis_angle_to_quaternion(axis * angle)
    if "xyaxes" in attrib:
        xyaxes = np.fromstring(attrib["xyaxes"], sep=" ")
        xaxis = xyaxes[:3] / np.linalg.norm(xyaxes[:3])
        zaxis = xyaxes[3:] / np.linalg.norm(xyaxes[:3])
        yaxis = np.cross(zaxis, xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
        rot_matrix = np.array([xaxis, yaxis, zaxis]).T
        return matrix_to_quaternion(rot_matrix)
    if "zaxis" in attrib:
        zaxis = np.fromstring(attrib["zaxis"], sep=" ")
        zaxis = zaxis / np.linalg.norm(zaxis)
        xaxis = np.cross(np.array([0, 0, 1]), zaxis)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
        rot_matrix = np.array([xaxis, yaxis, zaxis]).T
        return matrix_to_quaternion(rot_matrix)
    return np.array([1, 0, 0, 0])


@no_type_check
class MJCFLoader:
    """Class to load MJCF into SAPIEN."""

    def __init__(self, ignore_classes=["motor"]):
        self.fix_root_link = True
        """Fix the loader's dummy root link? False if the XML root body has a freejoint, else True (case-by-case)."""

        self.load_multiple_collisions_from_file = False
        self.multiple_collisions_decomposition = "none"
        self.multiple_collisions_decomposition_params = dict()

        # NOTE the following line is modified
        self.collision_is_visual = True
        self.revolute_unwrapped = False
        self.scale = 1.0

        self.scene: Optional[sapien.Scene] = None

        self.ignore_classes = ignore_classes

        self.density = 1000
        # NOTE the following line is modified
        self._link_material = dict()
        self._link_patch_radius = dict()
        self._link_min_patch_radius = dict()
        # self._link_density = dict()

        self._defaults: Dict[str, Element] = dict()
        self._assets = dict()
        self._materials = dict()
        self._textures: Dict[str, MJCFTexture] = dict()
        self._meshes: Dict[str, Element] = dict()

        self._link2builder: Dict[str, LinkBuilder] = dict()
        self._link2parent_joint: Dict[str, Any] = dict()
        self._group_count = 0

    def set_scene(self, scene):
        self.scene = scene
        return self

    @staticmethod
    def _pose_from_origin(origin, scale):
        origin[:3, 3] = origin[:3, 3] * scale
        return Pose(origin)

    def _build_geom(self, geom: Element, builder: Union[LinkBuilder, ActorBuilder], defaults):
        geom_defaults = defaults
        if "class" in geom.attrib:
            geom_class = geom.attrib["class"]
            ignore_geom = False
            for pattern in self.ignore_classes:
                if re.match(pattern, geom_class):
                    ignore_geom = True
                    break
            if ignore_geom:
                return
            if geom_class in self._defaults:
                geom_defaults = _merge_attrib(defaults, self._defaults[geom_class])
        if "geom" in geom_defaults:
            geom_attrib = _merge_attrib(geom_defaults["geom"], geom.attrib)
        else:
            geom_attrib = geom.attrib

        geom_name = geom_attrib.get("name", "")
        geom_type = geom_attrib.get("type", "sphere")
        if "mesh" in geom_attrib:
            geom_type = "mesh"

        geom_size = _parse_vec(geom_attrib, "size", [1.0, 1.0, 1.0]) * self.scale
        geom_pos = _parse_vec(geom_attrib, "pos", (0.0, 0.0, 0.0)) * self.scale
        geom_rot = _parse_orientation(geom_attrib, self._use_degrees, self._euler_seq)
        _parse_float(geom_attrib, "density", self.density)
        if "material" in geom_attrib:
            render_material = self._materials[geom_attrib["material"]]
        else:
            # use RGBA
            render_material = RenderMaterial(base_color=_parse_vec(geom_attrib, "rgba", [0.5, 0.5, 0.5, 1]))

        geom_density = _parse_float(geom_attrib, "density", 1000.0)
        # NOTE the following lines is modified
        link_name = builder.name
        if link_name in self._link_material:
            physx_material = PhysxMaterial(
                static_friction=self._link_material[link_name][0],
                dynamic_friction=self._link_material[link_name][1],
                restitution=self._link_material[link_name][2],
            )
        else:
            physx_material = None
        if link_name in self._link_patch_radius:
            patch_radius = self._link_patch_radius[link_name]
        else:
            patch_radius = 0
        if link_name in self._link_min_patch_radius:
            min_patch_radius = self._link_min_patch_radius[link_name]
        else:
            min_patch_radius = 0

        geom_group = _parse_int(geom_attrib, "group", 0)
        # See note at top of file for how we handle geom groups
        has_visual_body = False
        # NOTE the following line is modified
        if geom_group == 0 or geom_group == 2 or self.collision_is_visual:
            has_visual_body = True

        geom_contype = _parse_int(geom_attrib, "contype", 1)
        # See note at top of file for how we handle contype / objects without collisions
        has_collisions = True
        if geom_contype == 0:
            has_collisions = False

        t_visual2link = Pose(geom_pos, geom_rot)
        if geom_type == "sphere":
            if has_visual_body:
                builder.add_sphere_visual(t_visual2link, radius=geom_size[0], material=render_material)
            if has_collisions:
                builder.add_sphere_collision(
                    t_visual2link,
                    radius=geom_size[0],
                    material=physx_material,
                    density=geom_density,
                    patch_radius=patch_radius,
                    min_patch_radius=min_patch_radius,
                )
        elif geom_type in ["capsule", "cylinder", "box"]:
            if "fromto" in geom_attrib:
                geom_fromto = _parse_vec(geom_attrib, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

                start = np.array(geom_fromto[0:3]) * self.scale
                end = np.array(geom_fromto[3:6]) * self.scale
                # objects follow a line from start to end.

                # objects are default along x-axis and we rotate accordingly via axis angle.
                axis = (end - start) / np.linalg.norm(end - start)
                # TODO this is bugged
                angle = math.acos(np.dot(axis, np.array([1.0, 0.0, 0.0])))
                axis = np.cross(axis, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(axis) < 1e-3:
                    axis = np.array([1, 0, 0])
                else:
                    axis = axis / np.linalg.norm(axis)

                geom_pos = (start + end) * 0.5
                geom_rot = axis_angle_to_quaternion(axis * -angle)
                t_visual2link.set_p(geom_pos)
                t_visual2link.set_q(geom_rot)
                geom_radius = geom_size[0]
                geom_half_length = np.linalg.norm(end - start) * 0.5
            else:
                geom_radius = geom_size[0]
                geom_half_length = geom_size[1]
                # TODO (stao): oriented along z-axis for capsules whereas boxes are not?
                if geom_type in ["capsule", "cylinder"]:
                    t_visual2link = t_visual2link * Pose(q=axis_angle_to_quaternion(np.array([0.0, np.pi / 2, 0.0])))
            if geom_type == "capsule":
                if has_visual_body:
                    builder.add_capsule_visual(
                        t_visual2link,
                        radius=geom_radius,
                        half_length=geom_half_length,
                        material=render_material,
                        name=geom_name,
                    )
                if has_collisions:
                    builder.add_capsule_collision(
                        t_visual2link,
                        radius=geom_radius,
                        half_length=geom_half_length,
                        material=physx_material,
                        patch_radius=patch_radius,
                        min_patch_radius=min_patch_radius,
                        # name=geom_name,
                    )
            elif geom_type == "box":
                if has_visual_body:
                    builder.add_box_visual(
                        t_visual2link,
                        half_size=geom_size,
                        material=render_material,
                        name=geom_name,
                    )
                if has_collisions:
                    builder.add_box_collision(
                        t_visual2link,
                        half_size=geom_size,
                        material=physx_material,
                        patch_radius=patch_radius,
                        min_patch_radius=min_patch_radius,
                        # name=geom_name,
                    )
            elif geom_type == "cylinder":
                if has_visual_body:
                    builder.add_cylinder_visual(
                        t_visual2link,
                        radius=geom_radius,
                        half_length=geom_half_length,
                        material=render_material,
                        name=geom_name,
                    )
                if has_collisions:
                    builder.add_capsule_collision(
                        t_visual2link,
                        radius=geom_radius,
                        half_length=geom_half_length,
                        material=physx_material,
                        patch_radius=patch_radius,
                        min_patch_radius=min_patch_radius,
                        # name=geom_name
                    )

        elif geom_type == "plane":
            pass
        elif geom_type == "ellipsoid":
            pass
        elif geom_type == "cylinder":
            pass
        elif geom_type == "mesh":
            if has_visual_body:
                mesh_name = geom_attrib.get("mesh")
                builder.add_visual_from_file(
                    os.path.join(self._mesh_dir, self._meshes[mesh_name].get("file")),
                    scale=(self.scale, self.scale, self.scale),
                    material=render_material,
                )
        elif geom_type == "sdf":
            raise NotImplementedError("SDF geom type not supported at the moment")
        elif geom_type == "hfield":
            raise NotImplementedError("Height fields are not supported at the moment")

    def _build_link(self, body: Element, body_attrib, link_builder: LinkBuilder, defaults):
        """sets inertial, visual/collision shapes"""
        # inertial (NOTE: the following code is modified)
        inertial_elem = body.find("inertial")
        if inertial_elem is not None:
            pos = _parse_vec(inertial_elem.attrib, "pos", [0, 0, 0])
            quat = _parse_orientation(inertial_elem.attrib, self._use_degrees, self._euler_seq)
            mass = _parse_float(inertial_elem.attrib, "mass", 0)
            diaginertia = _parse_vec(inertial_elem.attrib, "diaginertia", [0, 0, 0])

            if not (np.allclose(diaginertia, [0, 0, 0]) or mass == 0):
                t_inertial2link = self._pose_from_origin(Pose(p=pos, q=quat).to_transformation_matrix(), self.scale)

                eigs = np.asarray(diaginertia)
                vecs = np.eye(3)

                assert all([x > 0 for x in eigs]), "invalid moment of inertia"

                t_inertia2inertial = np.eye(4)
                t_inertia2inertial[:3, :3] = vecs
                t_inertia2inertial = Pose(t_inertia2inertial)

                t_inertial2link = t_inertial2link * t_inertia2inertial
                scale3 = self.scale**3
                scale5 = self.scale**5
                link_builder.set_mass_and_inertia(mass * scale3, t_inertial2link, scale5 * eigs)

        # go through each geometry of the body
        for geo_count, geom in enumerate(body.findall("geom")):
            self._build_geom(geom, link_builder, defaults)

    def _parse_texture(self, texture: Element):
        """Parse MJCF textures for materials: https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-texture

        NOTE:
        - Procedural texture generation is currently not supported.
        - Different texture types are not really supported
        """
        name = texture.get("name")
        file = texture.get("file")
        self._textures[name] = MJCFTexture(
            name=name,
            type=texture.get("type"),
            rgb1=texture.get("rgb1"),
            rgb2=texture.get("rgb2"),
            file=os.path.join(self._mesh_dir, file) if file else None,
        )

    def _parse_material(self, material: Element):
        """Parse MJCF materials in asset to sapien render materials"""
        name = material.get("name")
        texture = None
        if material.get("texture") in self._textures:
            texture = self._textures[material.get("texture")]

        # NOTE: procedural texture generation is not supported; defaults from the MJCF asset-material spec.
        em_val = _parse_float(material.attrib, "emission", 0)
        rgba = np.array(_parse_vec(material.attrib, "rgba", [1, 1, 1, 1]))
        render_material = RenderMaterial(
            emission=[rgba[0] * em_val, rgba[1] * em_val, rgba[2] * em_val, 1],
            base_color=rgba,
            specular=_parse_float(material.attrib, "specular", 0),
            # TODO (stao): double check below 2 properties are right
            roughness=1 - _parse_float(material.attrib, "reflectance", 0),
            metallic=_parse_float(material.attrib, "shininess", 0.5),
        )
        if texture is not None and texture.file is not None:
            render_material.diffuse_texture = RenderTexture2D(filename=texture.file)
        self._materials[name] = render_material

    def _parse_mesh(self, mesh: Element):
        """Parse MJCF mesh data in asset"""
        # Vertex, normal, texcoord are not supported, file is required
        file = mesh.get("file")
        assert file is not None, (
            "Mesh file not provided. While Mujoco allows file to be optional, for loading into SAPIEN this is not optional"
        )
        name = mesh.get("name", os.path.splitext(file)[0])
        self._meshes[name] = mesh

    @property
    def _root_default(self):
        return self._defaults["__root__"]

    def _parse_default(self, node: Element, parent: Element):
        """Parse a MJCF default attribute: https://mujoco.readthedocs.io/en/stable/modeling.html#default-settings"""
        class_name = "__root__"
        if node.tag == "default":
            if "class" in node.attrib:
                class_name = node.attrib["class"]
            if parent is not None and "class" in parent.attrib:
                self._defaults[class_name] = deepcopy(self._defaults[parent.attrib["class"]])
            else:
                self._defaults[class_name] = {}
        for child in node:
            if child.tag == "default":
                self._parse_default(child, node)
            else:
                if child.tag in self._defaults[class_name]:
                    self._defaults[class_name][child.tag] = _merge_attrib(
                        self._defaults[class_name][child.tag], child.attrib
                    )
                else:
                    self._defaults[class_name][child.tag] = child.attrib

    def _parse_body(
        self,
        body: Element,
        parent: LinkBuilder,
        incoming_defaults: dict,
        builder: ArticulationBuilder,
    ):
        body_class = body.get("childclass")
        if body_class is None:
            defaults = incoming_defaults
        else:
            for pattern in self.ignore_classes:
                if re.match(pattern, body_class):
                    return
            defaults = _merge_attrib(incoming_defaults, self._defaults[body_class])

        if "body" in defaults:
            body_attrib = _merge_attrib(defaults["body"], body.attrib)
        else:
            body_attrib = body.attrib

        body_name = body_attrib["name"]
        body_pos = _parse_vec(body_attrib, "pos", (0.0, 0.0, 0.0))
        body_ori = _parse_orientation(body_attrib, use_degrees=self._use_degrees, euler_seq=self._euler_seq)

        body_pos *= self.scale
        body_pose = Pose(body_pos, q=body_ori)

        link_builder = parent

        joints = body.findall("joint")
        # if body has no joints, it is a fixed joint
        if len(joints) == 0:
            joints = [ET.Element("joint", attrib=dict(type="fixed"))]
        for i, joint in enumerate(joints):
            # note there can be multiple joints here. We create some dummy links to simulate that
            incoming_attributes = []
            if "joint" in defaults:
                incoming_attributes.append(defaults["joint"])
            if "class" in joint.attrib:
                incoming_attributes.append(self._defaults[joint.attrib["class"]]["joint"])
            incoming_attributes.append(joint.attrib)
            joint_attrib = _merge_attrib(dict(), incoming_attributes)

            # build the link
            link_builder = builder.create_link_builder(parent=link_builder)
            link_builder.set_joint_name(joint_attrib.get("name", ""))
            if i == len(joints) - 1:
                link_builder.set_name(f"{body_name}")
                # the last link is the "real" one, the rest are dummy links to support multiple joints acting on a link
                self._build_link(body, body_attrib, link_builder, defaults)
            else:
                link_builder.set_name(f"{body_name}_dummy_{i}")
            self._link2builder[link_builder.name] = link_builder

            joint_type = joint_attrib.get("type", "hinge")
            joint_pos = np.array(_parse_vec(joint_attrib, "pos", [0, 0, 0]))
            t_joint2parent = Pose()
            if i == 0:
                t_joint2parent = body_pose

            friction = _parse_float(joint_attrib, "frictionloss", 0)
            damping = _parse_float(joint_attrib, "damping", 0)

            # compute joint axis and relative transformations
            axis = _parse_vec(joint_attrib, "axis", [0.0, 0.0, 0.0])
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-3:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis /= axis_norm

            if abs(axis @ [1.0, 0.0, 0.0]) > 0.9:
                axis1 = np.cross(axis, [0.0, 0.0, 1.0])
                axis1 /= np.linalg.norm(axis1)
            else:
                axis1 = np.cross(axis, [1.0, 0.0, 0.0])
                axis1 /= np.linalg.norm(axis1)
            axis2 = np.cross(axis, axis1)
            t_axis2joint = np.eye(4)
            t_axis2joint[:3, 3] = joint_pos
            t_axis2joint[:3, 0] = axis
            t_axis2joint[:3, 1] = axis1
            t_axis2joint[:3, 2] = axis2
            t_axis2joint = Pose(t_axis2joint)
            t_axis2parent = t_joint2parent * t_axis2joint

            limited = joint_attrib.get("limited", "auto")
            if limited == "auto":
                if "range" in joint_attrib:
                    limited = True
                else:
                    limited = False
            elif limited == "true":
                limited = True
            else:
                limited = False

            # set the joint properties to create it
            if joint_type == "hinge":
                if limited:
                    joint_limits = _parse_vec(joint_attrib, "range", [0, 0])
                    if self._use_degrees:
                        joint_limits = np.deg2rad(joint_limits)
                    link_builder.set_joint_properties(
                        "revolute_unwrapped",
                        [joint_limits],
                        t_axis2parent,
                        t_axis2joint,
                        friction,
                        damping,
                    )
                else:
                    link_builder.set_joint_properties(
                        "revolute",
                        [[-np.inf, np.inf]],
                        t_axis2parent,
                        t_axis2joint,
                        friction,
                        damping,
                    )
            elif joint_type == "slide":
                if limited:
                    limits = [_parse_vec(joint_attrib, "range", [0, 0]) * self.scale]
                else:
                    limits = [[-np.inf, np.inf]]
                link_builder.set_joint_properties(
                    "prismatic",
                    limits,
                    t_axis2parent,
                    t_axis2joint,
                    friction,
                    damping,
                )
            elif joint_type == "fixed":
                link_builder.set_joint_properties(
                    "fixed",
                    [],
                    t_axis2parent,
                    t_axis2joint,
                    friction,
                    damping,
                )

        # Ensure adjacent links don't collide; dummy links (mujoco multi-joint) bypass SAPIEN's default check.
        if parent is not None:
            parent.collision_groups[2] |= 1 << (self._group_count)
            link_builder.collision_groups[2] |= 1 << (self._group_count)
            self._group_count += 1

        for child in body.findall("body"):
            self._parse_body(child, link_builder, defaults, builder)

    def _parse_constraint(self, constraint: Element):
        joint_elems = []
        for joint in constraint.findall("joint"):
            joint_elems.append(joint)
        return MimicJointRecord(
            joint_elems[0].attrib["joint"],
            joint_elems[1].attrib["joint"],
            1,
            0,
        )

    def _parse_mjcf(self, mjcf_string: str) -> Tuple[List[ArticulationBuilder], List[ActorBuilder], None]:
        """Helper function for self.parse"""
        xml: Element = ET.fromstring(mjcf_string.encode("utf-8"))
        # handle includes
        for include in xml.findall("include"):
            include_file = include.attrib["file"]
            with open(os.path.join(self.mjcf_dir, include_file), "r") as f:
                include_file_str = f.read()
                include_xml = ET.fromstring(include_file_str.encode("utf-8"))
            for child in include_xml:
                xml.append(child)

        self._use_degrees = True  # angles are in degrees by default
        self._mjcf_options = DEFAULT_MJCF_OPTIONS
        self._euler_seq = [1, 2, 3]  # XYZ by default

        ### Parse compiler options ###
        compiler = xml.find("compiler")
        if compiler is not None:
            self._use_degrees = compiler.attrib.get("angle", "degree").lower() == "degree"
            self._euler_seq = ["xyz".index(c) + 1 for c in compiler.attrib.get("eulerseq", "xyz").lower()]
            self._mesh_dir = compiler.attrib.get("meshdir", ".")
            self._mesh_dir = os.path.join(self.mjcf_dir, self._mesh_dir)

        ### Parse options/flags ###
        option = xml.find("option")
        if option is not None:
            for flag in option.findall("flag"):
                update_dict = dict()
                for k, v in flag.attrib.items():
                    update_dict[k] = True if v == "enable" else False
                self._mjcf_options.update(update_dict)

        ### Parse assets ###
        for asset in xml.findall("asset"):
            for texture in asset.findall("texture"):
                self._parse_texture(texture)
            for material in asset.findall("material"):
                self._parse_material(material)
            for mesh in asset.findall("mesh"):
                self._parse_mesh(mesh)

        ### Parse defaults ###
        for default in xml.findall("default"):
            self._parse_default(default, None)

        ### Parse Kinematic Trees / Articulations in World Body ###

        # NOTE (stao): assumes only one articulation; setups like Aloha 2 are technically 2 but can be treated as one.
        articulation_builders: List[ArticulationBuilder] = []
        actor_builders: List[ActorBuilder] = []
        for i, body in enumerate(xml.find("worldbody").findall("body")):
            # determine first if this body is really an articulation or a actor
            body.find("joint") is not None
            has_freejoint = body.find("freejoint") is not None
            # TODO (is it the case any <body> tag refers to an articulation?)
            if True:
                builder = self.scene.create_articulation_builder()
                articulation_builders.append(builder)
                dummy_root_link = builder.create_link_builder(None)
                dummy_root_link.name = f"dummy_root_{i}"
                self._parse_body(body, dummy_root_link, self._root_default, builder)

                # handle free joints
                fix_root_link = not has_freejoint
                if fix_root_link:
                    dummy_root_link.set_joint_properties(
                        type="fixed",
                        limits=None,
                        pose_in_parent=Pose(),
                        pose_in_child=Pose(),
                    )
            else:
                builder = self.scene.create_actor_builder()
                body_type = "dynamic" if has_freejoint else "static"
                actor_builders.append(builder)
                # TODO (stao): may not be correct. Does mujoco support multiple nested body tags to define geoms?
                for i, geom in enumerate(body.findall("geom")):
                    self._build_geom(geom, builder, self._root_default)
                    builder.set_name(geom.get("name", ""))
                builder.set_physx_body_type(body_type)

        ### Parse geoms in World Body ###
        # These can't have freejoints so they can't be dynamic
        for i, geom in enumerate(xml.find("worldbody").findall("geom")):
            builder = self.scene.create_actor_builder()
            actor_builders.append(builder)
            self._build_geom(geom, builder, self._root_default)
            builder.set_name(geom.get("name", ""))
            builder.set_physx_body_type("static")

        ### Parse contact and exclusions ###
        for contact in xml.findall("contact"):
            # TODO
            pass

        ### Parse equality constraints ###

        if not self._mjcf_options["contact"]:
            for actor in actor_builders:
                actor.collision_groups[2] |= 1 << 1
            for art in articulation_builders:
                for link in art.link_builders:
                    link.collision_groups[2] |= 1 << 1

        return articulation_builders, actor_builders, []

    def parse(self, mjcf_file: str, package_dir=None):
        """Parses a given MJCF file into articulation builders and actor builders and sensor configs"""
        self.package_dir = package_dir
        self.mjcf_dir = os.path.dirname(mjcf_file)

        with open(mjcf_file, "r") as f:
            mjcf_string = f.read()

        return self._parse_mjcf(mjcf_string)

    def load(self, mjcf_file: str, package_dir=None):
        """Parses a given mjcf .xml file and builds all articulations and actors"""
        articulation_builders, actor_builders, cameras = self.parse(mjcf_file, package_dir)

        articulations: List[PhysxArticulation] = []
        for b in articulation_builders:
            articulations.append(b.build())

        actors = []
        for b in actor_builders:
            actors.append(b.build())

        # TODO (stao): how does mjcf specify sensors?
        return articulations[0]

    # NOTE the following functions are modified
    def set_link_material(self, link_name, static_friction, dynamic_friction, restitution):
        self._link_material[link_name] = [
            static_friction,
            dynamic_friction,
            restitution,
        ]

    def set_link_patch_radius(self, link_name, patch_radius):
        self._link_patch_radius[link_name] = patch_radius

    def set_link_min_patch_radius(self, link_name, min_patch_radius):
        self._link_min_patch_radius[link_name] = min_patch_radius

    # TODO (stao): function to also load the scene in, and to load camera configs?
