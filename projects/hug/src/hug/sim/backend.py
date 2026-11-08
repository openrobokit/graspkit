"""In-process SAPIEN/PhysX G1 backend.

Body motors are implicit PD joint drives plus a gravity-comp feed-forward via ``qf``; hands are
position-driven with the 6 mimic joints tracking ``HAND_MIMIC_SCALE * proximal``. Every joint is
bound by name because SAPIEN reorders joints by tree traversal.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import sapien
import sapien.physx as physx
import trimesh
from scipy.spatial.transform import Rotation as R

from hug.control.config import ARM_KD, ARM_KP
from hug.sim.scene import (
    G1_JOINT_NAMES,
    HAND_MIMIC_FROM_URDF,
    HAND_MIMIC_SCALE,
    HAND_MIMIC_SUFFIXES,
    HAND_PROXIMAL_SUFFIXES,
    N_G1,
    STAND_ARM_QPOS,
    STAND_BASE_Z,
    STAND_HAND_QPOS,
    SapienPhysicsConfig,
    SceneConfig,
    object_collision_parts,
    object_mesh_path,
    object_scale,
    object_visual_mesh_path,
)
from hug.wbc.lower_body_policy import DEFAULT_LOWER_KD, DEFAULT_LOWER_KP, DEFAULT_LOWER_QPOS
from hug.wbc.pose_estimator import HEAD_QPOS, LinkPoseEstimator, motor_q_29_to_32

RENDER_W, RENDER_H = 640, 480  # head-camera render size
UNTEXTURED_OBJECT_COLOR = [0.85, 0.45, 0.12]  # render-only tint for objects without a textured visual

_FINGER_LINK_STEMS = [
    "hand_base_link",
    "thumb_proximal_base",
    "thumb_proximal",
    "thumb_intermediate",
    "thumb_distal",
    "index_proximal",
    "index_intermediate",
    "middle_proximal",
    "middle_intermediate",
    "ring_proximal",
    "ring_intermediate",
    "pinky_proximal",
    "pinky_intermediate",
]
FINGER_LINK_NAMES = [f"{p}{s}" for p in ("L_", "R_") for s in _FINGER_LINK_STEMS]
FOOT_LINK_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
HEAD_JOINT_NAMES = ["head_yaw_joint", "head_roll_joint", "head_pitch_joint"]  # position-driven scenery, not motors

# The URDF caps thumb_proximal_pitch at [0.14, 0.5]; the open thumb needs -0.4.
THUMB_PITCH_LIMIT = (-0.5, 0.5)

# Shapes sharing this group2 bit and group3 id never collide: all robot links and the table.
NOCOLLIDE_G2_BIT = 1 << 20
NOCOLLIDE_G3_ID = 1

# Optical (x-right, y-down, z-forward) → SAPIEN/ROS camera (x-fwd, y-left, z-up).
OPT_TO_ROS = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


class SimState(NamedTuple):
    motor_q: np.ndarray  # (29,) float32
    motor_qd: np.ndarray  # (29,) float32
    base_quat: np.ndarray  # (4,) wxyz, pelvis IMU
    gyro: np.ndarray  # (3,) pelvis-local angular velocity


def _pose(p, q_wxyz) -> sapien.Pose:
    return sapien.Pose(p=np.asarray(p, dtype=np.float64), q=np.asarray(q_wxyz, dtype=np.float64))


def _pose7(pose: sapien.Pose) -> np.ndarray:
    return np.array([*pose.p, *pose.q], dtype=np.float64)


def patched_urdf(urdf: Path) -> Path:
    """Write a sibling copy of ``urdf`` with its ``<mimic>`` tags stripped; return its path.

    SAPIEN makes mimic tags hard constraints, which the open thumb breaks."""
    out = urdf.with_name(f".{urdf.stem}_patched.urdf")
    tree = ET.parse(urdf)
    for joint in tree.getroot().findall("joint"):
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)
    tmp = out.parent / f"{out.name}.{os.getpid()}.tmp"
    tree.write(tmp)
    os.replace(tmp, out)  # atomic: concurrent readers never see a partial file
    return out


class SapienBackend:
    """In-process SAPIEN G1: PD-drive the 29 body motors, position the hands, substep PhysX.

    Tick: ``read_state()`` → ``set_command(...)`` → ``step()``; the root stays welded until :meth:`release_base`."""

    def __init__(
        self,
        urdf: Path,
        hz: float,
        scene_object: Optional[str] = None,
        physics: Optional[SapienPhysicsConfig] = None,
        scene: Optional[SceneConfig] = None,
    ) -> None:
        self.urdf = urdf
        self.physics = physics if physics is not None else SapienPhysicsConfig()
        self.scene_cfg = scene if scene is not None else SceneConfig()
        self._n_sub = max(1, round((1.0 / hz) / (1.0 / self.physics.sim_freq)))

        self._configure_physx()
        self._scene = sapien.Scene()
        self._scene.set_timestep(1.0 / self.physics.sim_freq)
        self._scene.add_ground(altitude=0.0)
        self._scene.set_ambient_light([0.45, 0.45, 0.45])  # render only; depth + seg drive the cloud
        self._scene.add_directional_light([-0.3, -0.5, -1.0], [1.6, 1.6, 1.6])
        self._scene.add_directional_light([0.6, 0.4, -0.4], [0.7, 0.7, 0.7])

        # ---- robot ----
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = False  # free base; pinned via _hold_root during warmup
        loader.load_multiple_collisions_from_file = False
        self.robot = loader.load(str(patched_urdf(urdf)))
        self.ndof = self.robot.dof
        for ln in self.robot.get_links():
            self._join_nocollide_club(ln)
        self._apply_finger_material()
        for j in self.robot.get_active_joints():
            j.set_armature(np.full(j.dof, self.physics.joint_armature))
        self.robot.set_solver_position_iterations(self.physics.solver_position_iterations)
        self.robot.set_solver_velocity_iterations(self.physics.solver_velocity_iterations)

        self._joints = self.robot.get_active_joints()
        self._jname_to_qi = {j.name: i for i, j in enumerate(self._joints)}
        self.body_qi = np.array([self._jname_to_qi[n] for n in G1_JOINT_NAMES], dtype=np.intp)
        self._bind_hands()

        # Body gains (lower policy + control.config) are applied on the first set_command after each reset.
        self._body_kp = np.concatenate([DEFAULT_LOWER_KP, ARM_KP])
        self._body_kd = np.concatenate([DEFAULT_LOWER_KD, ARM_KD])
        self._body_gains_applied = False

        self._imu_link = self._link("pelvis")  # the real G1's LowState IMU is the pelvis one; imu_link is on the torso
        self._torso_link = self._link("torso_link")
        self._pelvis_link = self._link("pelvis")
        self._foot_links = [self._link(n) for n in FOOT_LINK_NAMES]

        # Head pose the camera is posed from (and the 3-DoF head is driven to, when the URDF has one).
        self.head_qpos: np.ndarray = np.array(HEAD_QPOS, dtype=np.float64)
        self.head_qi = np.array(
            [self._jname_to_qi[n] for n in HEAD_JOINT_NAMES if n in self._jname_to_qi], dtype=np.intp
        )
        for qi in self.head_qi:
            self._joints[qi].set_drive_properties(
                stiffness=self.physics.head_drive_stiffness,
                damping=self.physics.head_drive_damping,
                force_limit=self.physics.head_drive_force_limit,
            )
        self._foot_rest_z = 0.0  # lower sole height with the stance planted; measured in reset()

        # ---- scene (table + object) ----
        self._object: Optional[sapien.Entity] = None
        self._table: Optional[sapien.Entity] = None
        self._table_top_z = self.scene_cfg.default_table_top_z
        self._object_seg_id = -1
        self._object_z_extent = 1.0
        self.last_cloud_z_fraction = 0.0  # completeness of the most recent capture
        self._spawn_xy = (self.scene_cfg.table_center[0], self.scene_cfg.table_center[1])
        self._spawn_yaw = self.scene_cfg.object_spawn_yaw
        self._camera: Optional[sapien.render.RenderCameraComponent] = None
        if scene_object is not None:
            self._build_scene(scene_object)

        self._tau_ff = np.zeros(N_G1)  # waist + arm gravity-comp feed-forward, applied via qf each substep
        self._weld_active = True
        self._stand_root_pose = _pose([0.0, 0.0, STAND_BASE_Z], [1.0, 0.0, 0.0, 0.0])
        self.reset()

    # ------------------------------------------------------------------ setup
    def _configure_physx(self) -> None:
        """Global PhysX defaults; must run before any shape or body exists."""
        shape_cfg = physx.PhysxShapeConfig()
        shape_cfg.contact_offset = self.physics.contact_offset
        shape_cfg.rest_offset = 0.0
        physx.set_shape_config(shape_cfg)
        body_cfg = physx.PhysxBodyConfig()
        body_cfg.solver_position_iterations = self.physics.solver_position_iterations
        body_cfg.solver_velocity_iterations = self.physics.solver_velocity_iterations
        physx.set_body_config(body_cfg)
        physx.set_default_material(self.physics.foot_ground_friction, self.physics.foot_ground_friction, 0.0)

    def _bind_hands(self) -> None:
        """Per side: bind proximal + mimic joints, widen the thumb pitch, set the drives, cache the limits."""
        self.hand_prox_qi: dict[str, np.ndarray] = {}
        self.hand_mimic_qi: dict[str, np.ndarray] = {}
        self._hand_prox_lim: dict[str, np.ndarray] = {}
        self._hand_mimic_lim: dict[str, np.ndarray] = {}
        for side, prefix in (("left", "L_"), ("right", "R_")):
            prox = [self._jname_to_qi.get(prefix + s) for s in HAND_PROXIMAL_SUFFIXES]
            mimic = [self._jname_to_qi.get(prefix + s) for s in HAND_MIMIC_SUFFIXES]
            if None in prox or None in mimic:
                continue
            self.hand_prox_qi[side] = np.array(prox, dtype=np.intp)
            self.hand_mimic_qi[side] = np.array(mimic, dtype=np.intp)
            # set_limits rebuilds the joint and drops its drive, so it must come first.
            self._joints[prox[1]].set_limits(np.array([THUMB_PITCH_LIMIT], dtype=np.float32))
            for qi in (*prox, *mimic):
                self._joints[qi].set_drive_properties(
                    stiffness=self.physics.hand_drive_stiffness,
                    damping=self.physics.hand_drive_damping,
                    force_limit=self.physics.hand_drive_force_limit,
                )
            self._hand_prox_lim[side] = np.array([self._joints[qi].get_limits()[0] for qi in prox])
            self._hand_mimic_lim[side] = np.array([self._joints[qi].get_limits()[0] for qi in mimic])

    def _link(self, name: str):
        for ln in self.robot.get_links():
            if ln.name == name:
                return ln
        raise RuntimeError(f"link not found: {name}")

    @staticmethod
    def _shapes(entity_owner):
        """Collision shapes of an articulation link or a built actor."""
        entity = getattr(entity_owner, "entity", entity_owner)
        for cls in (
            physx.PhysxArticulationLinkComponent,
            physx.PhysxRigidDynamicComponent,
            physx.PhysxRigidStaticComponent,
        ):
            body = entity.find_component_by_type(cls)
            if body is not None:
                return body.get_collision_shapes()
        raise RuntimeError(f"no physx body on {entity}")

    def _join_nocollide_club(self, entity_owner) -> None:
        """Robot links never collide with each other (as in the training env) nor with the table."""
        for shape in self._shapes(entity_owner):
            g = shape.get_collision_groups()
            shape.set_collision_groups([g[0], g[1], g[2] | NOCOLLIDE_G2_BIT, (g[3] & ~0xFFFF) | NOCOLLIDE_G3_ID])

    def _apply_finger_material(self) -> None:
        mat = physx.PhysxMaterial(
            static_friction=self.physics.finger_static_friction,
            dynamic_friction=self.physics.finger_dynamic_friction,
            restitution=self.physics.finger_restitution,
        )
        want = set(FINGER_LINK_NAMES)
        for ln in self.robot.get_links():
            if ln.name not in want:
                continue
            for shape in self._shapes(ln):
                shape.set_physical_material(mat)
                shape.set_patch_radius(self.physics.finger_patch_radius)
                shape.set_min_patch_radius(self.physics.finger_patch_radius)

    def _object_material(self) -> physx.PhysxMaterial:
        return physx.PhysxMaterial(
            self.physics.object_static_friction, self.physics.object_dynamic_friction, self.physics.object_restitution
        )

    def _build_scene(self, object_name: str) -> None:
        tb = self._scene.create_actor_builder()  # kinematic box: height is settable at runtime
        tb.add_box_collision(half_size=self.scene_cfg.table_half_size, material=self._object_material())
        tb.add_box_visual(half_size=self.scene_cfg.table_half_size, material=[0.6, 0.4, 0.2])
        self._table = tb.build_kinematic(name="table")
        self._join_nocollide_club(self._table)
        self._build_object(object_name)
        self._setup_camera()

    def _build_object(self, object_name: str) -> None:
        """(Re)build the target object: convex decomposition for collision, textured OBJ (if any) for the visual."""
        stl = object_mesh_path(object_name)
        scale = (object_scale(object_name),) * 3
        ob = self._scene.create_actor_builder()
        omat = self._object_material()
        # Decomposition, not a single hull: a hull of a concave object is fatter than the shape and gets shoved.
        for part in object_collision_parts(object_name):
            ob.add_convex_collision_from_file(filename=str(part), material=omat, scale=scale)
        visual = object_visual_mesh_path(object_name)
        if visual.name == "textured.obj":
            ob.add_visual_from_file(filename=str(visual), scale=scale)
        else:
            ob.add_visual_from_file(filename=str(visual), scale=scale, material=UNTEXTURED_OBJECT_COLOR)
        self._object = ob.build(name=object_name)
        self._object_body = self._object.find_component_by_type(physx.PhysxRigidDynamicComponent)
        self._object_body.set_mass(self.physics.object_mass)
        self._object_seg_id = int(self._object.per_scene_id)
        self._object_z_extent = float(trimesh.load(str(stl), force="mesh").extents[2] * object_scale(object_name))

    def swap_object(self, object_name: str) -> None:
        """Replace the target object in place (SAPIEN allows one rendered scene per process) and re-seat it."""
        self._scene.remove_actor(self._object)
        self._build_object(object_name)
        self.reset_object()

    def _setup_camera(self) -> None:
        """Mount the head camera on ``torso_link``; its pose is re-derived from the head URDF every capture."""
        self._estimator = LinkPoseEstimator()
        self._camera = self._scene.add_mounted_camera(
            name="head_camera",
            mount=self._torso_link.entity,
            pose=sapien.Pose(),
            width=RENDER_W,
            height=RENDER_H,
            fovy=np.deg2rad(self.scene_cfg.camera_fovy_deg),
            near=0.05,
            far=3.0,
        )
        self._align_camera()

    def _align_camera(self) -> np.ndarray:
        """Place the render camera where the estimator's ``camera_optical_frame`` is; returns ``world_T_camopt``."""
        q32 = motor_q_29_to_32(self.robot.get_qpos()[self.body_qi], self.head_qpos)
        w0_T_pelvis = self._estimator.estimate(q32, link_name="pelvis").matrix
        w0_T_cam = self._estimator.estimate(q32, link_name="camera_optical_frame").matrix
        pelvis_T_camopt = np.linalg.inv(w0_T_pelvis) @ w0_T_cam
        world_T_camopt = self._pelvis_link.get_pose().to_transformation_matrix() @ pelvis_T_camopt
        local = np.linalg.inv(self._torso_link.get_pose().to_transformation_matrix()) @ world_T_camopt
        local[:3, :3] = local[:3, :3] @ OPT_TO_ROS  # SAPIEN mount pose is ROS-convention
        assert self._camera is not None
        self._camera.set_local_pose(sapien.Pose(local))
        return world_T_camopt

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Stand pose, open hands, welded root; the object is re-seated if there is one."""
        self._weld_active = True
        stand = np.zeros(N_G1)
        stand[0:15] = DEFAULT_LOWER_QPOS
        stand[15:22] = STAND_ARM_QPOS
        stand[22:29] = STAND_ARM_QPOS
        qpos = np.zeros(self.ndof)
        qpos[self.body_qi] = stand
        mimic = HAND_MIMIC_SCALE * STAND_HAND_QPOS[HAND_MIMIC_FROM_URDF]
        for side in self.hand_prox_qi:
            qpos[self.hand_prox_qi[side]] = STAND_HAND_QPOS
            qpos[self.hand_mimic_qi[side]] = mimic
        qpos[self.head_qi] = self.head_qpos[: len(self.head_qi)]
        self.robot.set_root_pose(self._stand_root_pose)
        self.robot.set_root_linear_velocity(np.zeros(3))
        self.robot.set_root_angular_velocity(np.zeros(3))
        self.robot.set_qpos(qpos)
        self.robot.set_qvel(np.zeros(self.ndof))
        for side in self.hand_prox_qi:
            self._set_hand_drive(side, STAND_HAND_QPOS)
        self._set_head_drive()
        # Drop the body drives so the trailing step integrates the same on a reused backend as on a fresh one.
        for qi in self.body_qi:
            self._joints[qi].set_drive_properties(
                stiffness=0.0, damping=0.0, force_limit=self.physics.body_drive_force_limit
            )
        self._body_gains_applied = False
        for local_i, qi in enumerate(self.body_qi):
            self._joints[qi].set_drive_target(float(stand[local_i]))
        self._tau_ff[:] = 0.0
        self.robot.set_qf(np.zeros(self.ndof))  # PhysX keeps qf; clear the previous trial's torque too
        if self._object is not None:
            self._apply_table_height()
            self.reset_object()
        self._scene.step()
        self._foot_rest_z = self._lowest_foot_z()  # soles rest on the floor here by construction

    @property
    def base_welded(self) -> bool:
        return self._weld_active

    def release_base(self) -> None:
        self._weld_active = False

    def _lowest_foot_z(self) -> float:
        return min(float(ln.get_entity_pose().p[2]) for ln in self._foot_links)

    def _hold_root(self) -> None:
        """Pin the pelvis during warmup at whatever height keeps the lower foot planted.

        The balance policy squats during warmup; a fixed height would lift the feet, then drop the robot at release."""
        p = self._stand_root_pose.p
        z = float(self.robot.get_root_pose().p[2]) - (self._lowest_foot_z() - self._foot_rest_z)
        self.robot.set_root_pose(_pose([p[0], p[1], z], self._stand_root_pose.q))
        self.robot.set_root_linear_velocity(np.zeros(3))
        self.robot.set_root_angular_velocity(np.zeros(3))

    # ------------------------------------------------------------------ state
    def read_state(self) -> SimState:
        q = self.robot.get_qpos()[self.body_qi].astype(np.float32)
        qd = self.robot.get_qvel()[self.body_qi].astype(np.float32)
        pose = self._imu_link.get_pose()
        quat_wxyz = np.array(pose.q, dtype=np.float32)
        w_world = np.asarray(self._imu_link.get_angular_velocity(), dtype=np.float64)
        r_world_imu = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]])
        gyro = (r_world_imu.inv().apply(w_world)).astype(np.float32)
        return SimState(q, qd, quat_wxyz, gyro)

    def base_pose7(self) -> np.ndarray:
        """Pelvis (root) pose7 in the world frame."""
        return _pose7(self.robot.get_root_pose())

    def read_hand(self, side: str) -> np.ndarray:
        """Measured 6 proximal finger qpos for ``side``."""
        return self.robot.get_qpos()[self.hand_prox_qi[side]].astype(np.float64)

    def read_hand_mimic(self, side: str) -> np.ndarray:
        """Measured 6 mimic finger qpos (they are clamped to their limits, so not ``scale * proximal``)."""
        return self.robot.get_qpos()[self.hand_mimic_qi[side]].astype(np.float64)

    def read_head(self) -> np.ndarray:
        """Head ``[yaw, roll, pitch]``: measured when the URDF has a head, else the commanded ``head_qpos``."""
        if len(self.head_qi) == 0:
            return self.head_qpos.astype(np.float64)
        return self.robot.get_qpos()[self.head_qi].astype(np.float64)

    # ---------------------------------------------------------------- command
    def _set_hand_drive(self, side: str, q6: np.ndarray) -> None:
        """Drive the proximals to ``q6`` and the mimics to ``scale * proximal``, each clamped to its limits."""
        prox_lim = self._hand_prox_lim[side]
        mimic_lim = self._hand_mimic_lim[side]
        prox_target = np.clip(q6, prox_lim[:, 0], prox_lim[:, 1])
        mimic_target = np.clip(HAND_MIMIC_SCALE * q6[HAND_MIMIC_FROM_URDF], mimic_lim[:, 0], mimic_lim[:, 1])
        for local_i, qi in enumerate(self.hand_prox_qi[side]):
            self._joints[qi].set_drive_target(float(prox_target[local_i]))
        for local_i, qi in enumerate(self.hand_mimic_qi[side]):
            self._joints[qi].set_drive_target(float(mimic_target[local_i]))

    def _set_head_drive(self) -> None:
        for local_i, qi in enumerate(self.head_qi):
            self._joints[qi].set_drive_target(float(self.head_qpos[local_i]))

    def _apply_body_gains(self) -> None:
        for local_i, qi in enumerate(self.body_qi):
            self._joints[qi].set_drive_properties(
                stiffness=float(self._body_kp[local_i]),
                damping=float(self._body_kd[local_i]),
                force_limit=self.physics.body_drive_force_limit,
            )
        self._body_gains_applied = True

    def set_command(
        self,
        qdes15: np.ndarray,
        arm_qpos: np.ndarray,
        arm_tau_ff: np.ndarray,
        hand_qpos: np.ndarray,
        waist_tau_ff: np.ndarray,
    ) -> None:
        """Write one tick: legs+waist targets (15), arm targets (14), arm and waist feed-forward (14, 3), hands (12)."""
        if not self._body_gains_applied:
            self._apply_body_gains()
        target29 = np.concatenate([qdes15, arm_qpos])
        for local_i, qi in enumerate(self.body_qi):
            self._joints[qi].set_drive_target(float(target29[local_i]))
        self._tau_ff[:] = 0.0
        # not while welded: against a pinned root the balance policy's legs creep and the feet slide apart
        self._tau_ff[12:15] = 0.0 if self._weld_active else waist_tau_ff
        self._tau_ff[15:29] = arm_tau_ff
        hand_qpos = np.asarray(hand_qpos, dtype=np.float64)
        for side, q6 in (("left", hand_qpos[:6]), ("right", hand_qpos[6:])):
            if side in self.hand_prox_qi:
                self._set_hand_drive(side, q6)
        self._set_head_drive()

    def step(self) -> None:
        """Substep PhysX; drives supply the PD torque, ``qf`` the feed-forward; root pinned while welded."""
        qf = np.zeros(self.ndof)
        qf[self.body_qi] = self._tau_ff
        for _ in range(self._n_sub):
            self.robot.set_qf(qf)
            if self._weld_active:
                self._hold_root()
            self._scene.step()

    # ------------------------------------------------------------ scene / obj
    def object_pose7(self) -> np.ndarray:
        assert self._object is not None, "no scene object"
        return _pose7(self._object.get_pose())

    @property
    def table_pos(self) -> np.ndarray:
        cx, cy, _ = self.scene_cfg.table_center
        return np.array([cx, cy, self._table_top_z - self.scene_cfg.table_half_size[2]], dtype=np.float64)

    @property
    def table_half_size(self) -> tuple[float, float, float]:
        return self.scene_cfg.table_half_size

    @property
    def table_top_z(self) -> float:
        return float(self._table_top_z)

    def _apply_table_height(self) -> None:
        assert self._table is not None, "no scene table"
        self._table.set_pose(_pose(self.table_pos, [1.0, 0.0, 0.0, 0.0]))

    def set_table_height(self, top_z: float) -> None:
        self._table_top_z = float(top_z)
        self._apply_table_height()
        self.reset_object()

    def randomize_object(
        self, rng: Optional[np.random.Generator] = None, *, ensure_visible: bool = True, tries: int = 12
    ) -> np.ndarray:
        """Draw a new spawn ``(x, y, yaw)`` within the scene's bounds and seat the object there.

        With ``ensure_visible`` the draw is retried (best of ``tries`` kept) until the camera sees enough of it."""
        rng = np.random.default_rng() if rng is None else rng
        cx, cy, _ = self.scene_cfg.table_center
        best: Optional[tuple[float, float, float]] = None
        best_frac = -1.0
        for _ in range(max(1, tries)):
            self._spawn_xy = (
                cx + float(rng.uniform(*self.scene_cfg.random_x_offset)),
                cy + float(rng.uniform(*self.scene_cfg.random_y_offset)),
            )
            self._spawn_yaw = float(rng.uniform(-np.pi, np.pi))
            self.reset_object()
            if not ensure_visible:
                break
            self.capture_object_pc()
            if self.last_cloud_z_fraction > best_frac:
                best, best_frac = (*self._spawn_xy, self._spawn_yaw), self.last_cloud_z_fraction
            if best_frac >= self.scene_cfg.min_cloud_z_fraction:
                break
        else:
            assert best is not None
            self._spawn_xy, self._spawn_yaw = (best[0], best[1]), best[2]
            self.reset_object()
        return np.array([*self._spawn_xy, self._spawn_yaw])

    def reset_object(self) -> None:
        """Re-seat the object at its spawn pose, just above the table, at rest."""
        x, y = self._spawn_xy
        quat = R.from_euler("z", self._spawn_yaw).as_quat()  # xyzw
        assert self._object is not None, "no scene object"
        self._object.set_pose(
            _pose(
                [x, y, self._table_top_z + self.scene_cfg.object_drop_clearance],
                [quat[3], quat[0], quat[1], quat[2]],
            )
        )
        self._object_body.set_linear_velocity(np.zeros(3))
        self._object_body.set_angular_velocity(np.zeros(3))

    def capture_object_pc(self) -> np.ndarray:
        """Render the head camera; object points in ``camera_optical_frame``, or ``(0, 3)`` if not visible."""
        if self._object is None:
            raise RuntimeError("capture_object_pc requires scene_object at construction")
        world_T_camopt = self._align_camera()
        assert self._camera is not None
        self._scene.update_render()
        self._camera.take_picture()
        pos = self._camera.get_picture("Position")  # (H, W, 4), OpenGL camera frame
        seg = self._camera.get_picture("Segmentation")[:, :, 1]  # actor per_scene_id
        xyz = pos[:, :, :3].reshape(-1, 3)
        mask = (seg.reshape(-1) == self._object_seg_id) & ((-xyz[:, 2]) > 1e-4)  # OpenGL: forward = -z
        if int(mask.sum()) < self.scene_cfg.min_object_pixels:
            self.last_cloud_z_fraction = 0.0
            return np.zeros((0, 3))
        cam2world = self._camera.get_model_matrix()
        world = (cam2world[:3, :3] @ xyz[mask].T).T + cam2world[:3, 3]
        self.last_cloud_z_fraction = float(np.ptp(world[:, 2]) / self._object_z_extent)
        camopt_T_world = np.linalg.inv(world_T_camopt)
        return ((camopt_T_world[:3, :3] @ world.T).T + camopt_T_world[:3, 3]).astype(np.float64)
