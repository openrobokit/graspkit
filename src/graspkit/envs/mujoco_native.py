"""Closed-loop MuJoCo tabletop grasping, one model per scene, with Isaac's robot export, drives, cameras and success."""

import copy
import hashlib
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # mujoco picks its GL backend on first import
import mujoco
import numpy as np
import torch
import trimesh
from robokit.xform.numpy import matrix_to_quaternion, quaternion_multiply
from robokit.xform.utils import coord_conversion
from torch.types import Device

from graspkit.configs.experiment_config import EnvConfig, RendererConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.constants import GRASPKIT_HOME, LOG_ROOT
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs.base import BaseEnv, Rollout
from graspkit.isaac.job import export_robot
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.point_cloud_utils import crop_and_resample_point_cloud

SAPIEN_TO_GL = coord_conversion("sapien", "opengl")
"""Camera axes: SAPIEN looks along +x with +z up, MuJoCo (OpenGL) along -z with +y up."""
TIP_FRICTION, TIP_TORSION, DEFAULT_FRICTION = 2.0, 0.08, 0.3
"""ManiSkill's fingertip/object friction, tip torsional friction times the 0.04 m min patch radius, default friction."""


class MjNativeEnv(BaseEnv):
    """Fixed-base tabletop envs, one MuJoCo model per scene, with robot cameras; task settings as in ManiSkill/Isaac."""

    def __init__(self, cfg: EnvConfig, robot_config: RobotConfig, device: Device, renderer_cfg: RendererConfig):
        super().__init__()
        self.cfg, self.renderer_cfg, self.device = cfg, renderer_cfg, torch.device(device)
        settings = self.settings = cfg.mj_native_env_config
        self.substeps = round(1 / (cfg.control_freq * settings.timestep))
        assert abs(self.substeps * settings.timestep * cfg.control_freq - 1) < 1e-6, "Control step must be whole steps."
        robot = self.robot = export_robot(robot_config, cfg.cameras)
        tree = robot["tree"]
        # MuJoCo hulls each mesh file and names it by stem: write content-named OBJ pieces, split as SAPIEN splits.
        cache = LOG_ROOT / "mujoco/meshes"
        cache.mkdir(parents=True, exist_ok=True)
        for link in tree.getroot().iter("link"):
            for element in [child for child in link if child.find("geometry/mesh") is not None]:
                source = Path(element.find("geometry/mesh").get("filename"))
                digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
                link.remove(element)
                parts = trimesh.load(source, force="scene").dump()
                if element.tag == "collision" and source.suffix.lower() != ".glb":  # flat slivers have no hull
                    parts = trimesh.util.concatenate(parts).split(only_watertight=False)
                    parts = [
                        part for part in parts if np.linalg.matrix_rank(part.vertices - part.vertices.mean(0)) == 3
                    ]
                for index, part in enumerate(parts):
                    path = cache / f"{digest}_{element.tag}_{index}.obj"
                    if not path.exists():
                        partial = path.with_suffix(f".{os.getpid()}.obj")
                        trimesh.Trimesh(part.vertices, part.faces, process=False).export(partial)
                        partial.replace(path)
                    piece = copy.deepcopy(element)
                    piece.find("geometry/mesh").set("filename", str(path))
                    if element.tag == "visual" and part.visual.kind is not None:  # plain STLs keep the URDF color
                        textured = part.visual.kind == "texture"  # robot GLBs carry flat PBR colors
                        rgba = (
                            part.visual.material.main_color if textured else part.visual.vertex_colors.mean(0)
                        ) / 255
                        for material in piece.findall("material"):
                            piece.remove(material)
                        material = ET.SubElement(piece, "material", name=path.stem)
                        ET.SubElement(material, "color", rgba=" ".join(f"{value:.4f}" for value in rgba))
                    link.append(piece)
        options = dict(strippath="false", fusestatic="false", discardvisual="false")
        ET.SubElement(ET.SubElement(tree.getroot(), "mujoco"), "compiler", **options)
        spec = self.spec = mujoco.MjSpec.from_string(ET.tostring(tree.getroot(), encoding="unicode"))
        spec.compiler.balanceinertia = True  # PhysX accepts a few URDF inertias that violate the triangle inequality
        for mesh in spec.meshes:  # links carry URDF inertials; thin visual pieces have no volume
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        spec.option.timestep, spec.option.impratio = settings.timestep, settings.impratio
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER  # see the drive armature below
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        if mujoco.__version__.startswith(("3.4.", "3.5.", "3.6.", "3.7.")):  # 3.8 turned multi-CCD on by default
            spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD  # a resting hull needs several contacts
        spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.4] * 3, [0.6] * 3
        for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):  # robot links: no gravity, no self-collision
            body.gravcomp = 1
            for geom in body.geoms:
                if geom.contype:  # collision geoms are hidden and touch only the table and the object
                    tip = body.name in robot["tip_links"]
                    geom.contype, geom.conaffinity, geom.group = 2, 0, 3
                    geom.solref, geom.solimp, geom.condim = settings.solref, settings.solimp, 4 if tip else 3
                    geom.friction[:2] = (TIP_FRICTION, TIP_TORSION) if tip else (DEFAULT_FRICTION, 0)
        # One-step damping armature mimics PhysX implicit drives and steadies Euler; implicit integrators fling objects.
        for name in robot["joint_names"]:
            stiffness, damping, limit = (1e3, 50, 50) if name in robot["hand_joints"] else (1e5, 1e4, 500)
            joint = spec.joint(name)
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE  # drives cap force, not URDF efforts
            joint.armature = settings.timestep * damping
            actuator = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            actuator.set_to_position(kp=stiffness, kv=damping)
            actuator.forcelimited, actuator.forcerange = mujoco.mjtLimited.mjLIMITED_TRUE, [-limit, limit]
        for index, camera in enumerate(robot["cameras"]):
            k, size = np.asarray(camera["intrinsic"]), np.asarray([camera["width"], camera["height"]])
            (spec.worldbody if camera["mount"] is None else spec.body(camera["mount"])).add_camera(
                name=f"camera{index}",
                pos=camera["pose"][:3],
                quat=quaternion_multiply(np.asarray(camera["pose"][3:]), matrix_to_quaternion(SAPIEN_TO_GL.T)),
                resolution=size,
                sensor_size=size,
                focal_pixel=[k[0, 0], k[1, 1]],
                principal_pixel=k[:2, 2] - size / 2,
            )
        self.pool = ThreadPoolExecutor(settings.num_threads)
        self._scenes, self._pieces, self.renderers = [], {}, []

    def close(self):
        self.pool.shutdown()
        for renderer in self.renderers:
            renderer.close()

    def _full_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Pruned chain qpos (..., D) -> every URDF joint, expanding mimics and parked branches."""
        return qpos @ self.robot["coefficients"].T + self.robot["bias"]

    def reset(
        self,
        scene: SceneDescription | list[SceneDescription] | None = None,
        init_qpos: torch.Tensor | None = None,
        root_pose: torch.Tensor | None = None,
        camera_pose: torch.Tensor | None = None,
    ):
        """One model per scene in `scene` (a scene or a list); `None` reuses the previous scenes."""
        assert camera_pose is None, "MuJoCo observes through the robot's own cameras."
        if scene is not None:
            self._scenes = scene if isinstance(scene, list) else [scene]
        n = len(self._scenes)
        qpos = self.robot["init_qpos"] if init_qpos is None else init_qpos.detach().cpu().numpy()
        qpos = self._full_qpos(np.broadcast_to(qpos.reshape(-1, qpos.shape[-1]), (n, qpos.shape[-1])))
        root = (self.robot["root_pose"] if root_pose is None else root_pose.detach().cpu().numpy()).reshape(-1, 7)[0]
        for renderer in self.renderers:
            renderer.close()
        self.models, self.datas, self.renderers = [], [], []
        for scene, target in zip(self._scenes, qpos):
            spec = self.spec.copy()
            spec.worldbody.first_body().pos, spec.worldbody.first_body().quat = root[:3], root[3:]
            table = scene.boxes["table"]
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.asarray(table.size) / 2,
                pos=table.position,
                quat=table.quaternion,
                rgba=[0.8, 0.47, 0.29, 1],  # the mean color of ManiSkill's wooden table under this lighting
                contype=1,
                conaffinity=3,
                solref=self.settings.solref,
                solimp=self.settings.solimp,
                friction=[DEFAULT_FRICTION, 0, 0],
            )
            mesh = scene.meshes[scene.target_actor_id]
            key = (mesh.collision_mesh_path, tuple(mesh.scales))
            if key not in self._pieces:  # convex pieces, one hull each, with scale baked in
                source = trimesh.load(GRASPKIT_HOME / mesh.collision_mesh_path, process=False, force="mesh")
                self._pieces[key] = [
                    (part.vertices * mesh.scales, part.faces) for part in source.split(only_watertight=False)
                ]
            body = spec.worldbody.add_body(name="target", pos=mesh.position, quat=mesh.quaternion)
            body.add_freejoint()
            for index, (vertices, faces) in enumerate(self._pieces[key]):
                spec.add_mesh(name=f"target_{index}", uservert=vertices.ravel(), userface=faces.ravel())
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=f"target_{index}",
                    rgba=[0.6, 0.6, 0.6, 1],  # SAPIEN's default for untextured meshes
                    contype=1,
                    conaffinity=3,
                    solref=self.settings.solref,
                    solimp=self.settings.solimp,
                    friction=[(TIP_FRICTION + DEFAULT_FRICTION) / 2, 0, 0],  # PhysX averages, MuJoCo takes the larger
                )
            model = spec.compile()
            model.body_mass[-1] = self.cfg.object_mass  # SAPIEN overrides the mass, keeping the density-1000 inertia
            model.vis.map.znear, model.vis.map.zfar = 0.005 / model.stat.extent, 5 / model.stat.extent
            model.vis.quality.offsamples = 0  # multisampled depth and segmentation read off-center samples
            data = mujoco.MjData(model)
            mujoco.mj_setConst(model, data)
            self.joints = [model.joint(name).qposadr[0] for name in self.robot["joint_names"]]
            data.qpos[self.joints] = data.ctrl[:] = target
            mujoco.mj_forward(model, data)
            self.models.append(model)
            self.datas.append(data)
        model = self.models[0]
        self.policy_joints = [self.robot["joint_names"].index(name) for name in self.robot["policy_joint_names"]]
        self.contact_bodies = [model.body(name).id for name in self.robot["contact_links"]]
        self.robot_root, self.target = model.body(self.spec.worldbody.first_body().name).id, model.body("target").id
        self.initial_z = np.asarray([data.qpos[-5] for data in self.datas])  # the free joint is the last one

    def get_qpos(self) -> torch.Tensor:
        qpos = np.stack([data.qpos[self.joints] for data in self.datas])[:, self.policy_joints]
        return torch.as_tensor(qpos, dtype=torch.float32, device=self.device)

    def step_qpos(self, q: torch.Tensor):
        targets = self._full_qpos(q.detach().cpu().numpy().reshape(len(self.datas), -1))
        for data, target in zip(self.datas, targets):
            data.ctrl[:] = target
        list(self.pool.map(lambda i: mujoco.mj_step(self.models[i], self.datas[i], self.substeps), range(len(targets))))

    def get_object_pose(self) -> torch.Tensor:
        return torch.as_tensor(
            np.stack([data.qpos[-7:] for data in self.datas]), dtype=torch.float32, device=self.device
        )

    def set_object_pose(self, pose: torch.Tensor):
        for data, target in zip(self.datas, pose.detach().cpu().numpy()):
            data.qpos[-7:], data.qvel[-6:] = target, 0

    def get_info(self) -> dict[str, torch.Tensor]:
        """Lifted past `target_lift_dist` while at least two contact links press the object with 0.5 N or more."""
        contacts, force = [], np.zeros(6)
        for model, data in zip(self.models, self.datas):
            mujoco.mj_forward(model, data)
            forces = np.zeros((len(self.contact_bodies), 3))
            bodies = model.geom_bodyid[data.contact.geom]
            for c in np.flatnonzero((bodies == self.target).any(axis=1)):
                link = bodies[c, 1] if bodies[c, 0] == self.target else bodies[c, 0]
                if link in self.contact_bodies:
                    mujoco.mj_contactForce(model, data, c, force)  # contact frame; geom1 pushes geom2 along the normal
                    world = data.contact.frame[c].reshape(3, 3).T @ force[:3]
                    forces[self.contact_bodies.index(link)] += world if bodies[c, 1] == self.target else -world
            contacts.append((np.linalg.norm(forces, axis=1) >= 0.5).sum())
        lift = np.asarray([data.qpos[-5] for data in self.datas]) - self.initial_z
        lifted, grasped = lift > self.cfg.target_lift_dist, np.asarray(contacts) >= 2
        info = {"success": lifted & grasped, "is_lifted": lifted, "is_strict_grasped": grasped, "lift_dz": lift}
        return {key: torch.as_tensor(value, device=self.device) for key, value in info.items()}

    def observe(self) -> dict[str, torch.Tensor]:
        """Per scene/camera: RGB and robot-free, target-flagged world points, cropped and resampled like ManiSkill."""
        cameras = self.robot["cameras"]
        height, width = cameras[0]["height"], cameras[0]["width"]
        if not self.renderers:
            self.renderers = [mujoco.Renderer(model, height, width) for model in self.models]
        v, u = np.mgrid[:height, :width] + 0.5
        rgb, clouds, targets, poses = [], [], [], []
        for model, data, renderer in zip(self.models, self.datas, self.renderers):
            mujoco.mj_forward(model, data)
            for index, camera in enumerate(cameras):
                renderer.update_scene(data, camera=f"camera{index}")
                rgb.append(renderer.render())
                renderer.enable_depth_rendering()
                depth = renderer.render()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                geom = renderer.render()[..., 0]
                renderer.disable_segmentation_rendering()
                k, rotation = np.asarray(camera["intrinsic"]), data.cam_xmat[index].reshape(3, 3)
                local = (
                    np.stack(((u - k[0, 2]) / k[0, 0], (k[1, 2] - v) / k[1, 1], -np.ones_like(u)), -1)
                    * depth[..., None]
                )
                root = model.body_rootid[np.where(geom >= 0, model.geom_bodyid[geom], 0)]
                visible = (geom >= 0) & (root != self.robot_root)
                clouds.append(np.concatenate((local @ rotation.T + data.cam_xpos[index], visible[..., None]), -1))
                targets.append(root == self.target)
                poses.append(np.concatenate((data.cam_xpos[index], matrix_to_quaternion(rotation @ SAPIEN_TO_GL))))
        n, views = len(self.models), len(cameras)
        cloud = torch.from_numpy(np.stack(clouds)).float().reshape(n * views, -1, 4)
        tables = torch.tensor([scene.boxes["table"].position for scene in self._scenes]).repeat_interleave(views, 0)
        points = crop_and_resample_point_cloud(
            {
                "xyzw": cloud,
                "rgb": torch.zeros_like(cloud[..., :3]),
                "segmentation": torch.from_numpy(np.stack(targets)).float().reshape(n * views, -1, 1),
            },
            torch.tensor(self.renderer_cfg.points_crop_bounds) + tables[:, :, None],
            self.renderer_cfg.num_points,
        )
        return {
            "rgb": torch.from_numpy(np.stack(rgb)).reshape(n, views, height, width, 3),
            "xyz": points["xyz"].reshape(n, -1, 3),
            "seg": points["segmentation"].reshape(n, -1, 1),
            "seg_id": torch.ones(n),
            "camera_poses": torch.from_numpy(np.stack(poses)).float().reshape(n, views, 7),
        }

    def get_rgb(self) -> np.ndarray:
        return self.observe()["rgb"][0, 0].numpy()

    def get_pointcloud(self, mask: np.ndarray) -> torch.Tensor:
        raise NotImplementedError("MuJoCo removes the robot by segmentation; use observe().")

    def evaluate(self, batch: GraspKitData, record: bool = False) -> dict[str, torch.Tensor]:
        """Roll out `batch.full_traj_qpos` (B, K, T, D) in `num_envs` chunks; deterministic, so one seed decides."""
        assert batch.full_traj_qpos is not None and batch.scene_descriptions is not None
        assert not self.cfg.enable_perturbation, "MuJoCo evaluates without perturbation."
        trajectories = batch.full_traj_qpos.detach().cpu()
        if trajectories.dim() == 3:
            trajectories = trajectories[:, None]
        b, k, t, d = trajectories.shape
        trajectories = trajectories.reshape(b * k, t, d)
        scenes = [scene for scene in batch.scene_descriptions for _ in range(k)]
        self.last_rollouts, results = [], []
        for start in range(0, b * k, self.cfg.num_envs):
            stop = min(start + self.cfg.num_envs, b * k)
            self.reset(scenes[start:stop], init_qpos=trajectories[start:stop, 0])
            measured, poses = [], []
            for step in range(t):
                self.step_qpos(trajectories[start:stop, step])
                if record:
                    measured.append(self.get_qpos().cpu().numpy())
                    poses.append(self.get_object_pose().cpu().numpy())
            results.append({key: value.cpu().numpy() for key, value in self.get_info().items()})
            for offset, index in enumerate(range(start, stop) if record else []):
                self.last_rollouts.append(
                    Rollout(
                        qpos=np.stack([qpos[offset] for qpos in measured]),
                        obj_pose=np.stack([pose[offset] for pose in poses]),
                        scene_id=batch.scene_ids[index // k],
                        grasp_index=index % k,
                        success=bool(results[-1]["success"][offset]),
                    )
                )
        return {
            key: torch.as_tensor(np.concatenate([result[key] for result in results]), device=self.device).reshape(b, k)
            for key in results[0]
        }
