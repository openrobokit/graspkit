"""MuJoCo hand-object grasp env; re-run the golden test after any physics/solver setup or call-order change."""

import os

import mujoco
import numpy as np
import trimesh

from graspkit.mujoco.rot_np import interpolate_pose, interpolate_qpos


class MjDexGraspEnv:
    hand_prefix: str = "child-"

    def __init__(
        self,
        obj_path: str,
        obj_scale: float,
        obj_density: float,
        hand_xml_path: str,
        hand_mocap: bool,
        exclude_table_contact: list[str] | None,
        friction_coef: float,
        table_box: tuple | None,
        record: bool = False,
        timestep: float = 0.004,
        contact_solref_timeconst: float = 0.005,
    ):
        # record: log per-step hand qpos + object pose for replay (persisted as ReplayData).
        self.hand_mocap = hand_mocap
        self.record = record
        self._solref_timeconst = contact_solref_timeconst  # rigid-contact solref[0]; larger = softer
        self.sim_steps = 0  # total mj_step physics steps taken (for eval-speed benchmarking)
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.path.dirname(hand_xml_path)
        # 0.004 for mocap; a smaller step keeps the grip stable under contact (else stiff servos blow up the solver).
        self.spec.option.timestep = timestep
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
        # Dense hand-object contact can exceed MuJoCo's default arena.
        self.spec.memory = 128 << 20

        self._add_hand(hand_xml_path, hand_mocap)
        self._add_object(obj_path, obj_scale, obj_density, table_box)
        self._set_friction(friction_coef)
        self.spec.add_key()
        if exclude_table_contact is not None:
            for body_name in exclude_table_contact:
                self.spec.add_exclude(bodyname1="world", bodyname2=f"{self.hand_prefix}{body_name}")

        # Get ready for simulation
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # For ctrl
        qpos2ctrl_matrix = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            qpos2ctrl_matrix,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., :-6]

        # Per-step observation logs, populated in control_hand_step when record=True.
        self.hand_qpos_log = []
        self.obj_pose_log = []
        return

    def _add_hand(self, xml_path: str, mocap_base: bool):
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            # solimp/solref from the Shadow Hand xml: larger force at smaller penetration (more rigid, less soft).
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [self._solref_timeconst, 1]

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(child_spec.worldbody, self.hand_prefix, "")
        # Add freejoint and mocap of hand root
        if mocap_base:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=[0.9, 0.95, 0.001, 0.5, 2],
                data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            )
        return

    def _add_object(self, obj_path: str, obj_scale: float, obj_density: float, table_box: tuple | None):
        # table_box = (pos, half_size, quat) of the scene's finite support table in the mjcf/base frame.
        if table_box is not None:
            pos, half_size, quat = table_box
            # bit 2 only: object carries bits 1|2, so the table supports it (bit 2) but never touches the hand (bit 1).
            self.spec.worldbody.add_geom(
                name="object_support_table",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(pos),
                quat=list(quat),
                size=list(half_size),
                contype=2,
                conaffinity=2,
            )

        obj_body = self.spec.worldbody.add_body(name="object")
        obj_body.add_freejoint(name="obj_freejoint")
        # .glb scene geoms are convex pieces; add each as inline mesh+geom after its mesh (inertia sums in add order).
        if obj_path.endswith(".glb"):
            for i, part in enumerate(trimesh.load(obj_path, process=False).dump(concatenate=False)):
                name = f"convex_piece_{i}"
                mesh = self.spec.add_mesh(
                    name=name, uservert=part.vertices.flatten().tolist(), userface=part.faces.flatten().tolist()
                )
                mesh.scale = [obj_scale, obj_scale, obj_scale]
                obj_body.add_geom(
                    name=f"object_visual_{i}",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=name,
                    density=0,
                    contype=0,
                    conaffinity=0,
                )
                obj_body.add_geom(
                    name=f"object_collision_{i}",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=name,
                    density=obj_density,
                    contype=3,  # bits 1|2: collide with both the hand (bit 1) and the support floor (bit 2)
                    conaffinity=3,
                )
            return
        # obj_path holds the convex-piece .obj files; keep os.listdir order (MuJoCo sums geom inertia in add order):
        # - convex_piece_*.obj or coacd_convex_piece_*.obj (both layouts work)
        # - excluding the merged decomposed.obj
        for file in os.listdir(obj_path):
            if "convex_piece_" not in file or not file.endswith(".obj"):
                continue
            file_path = os.path.join(obj_path, file)
            mesh_name = file.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")

            self.spec.add_mesh(
                name=mesh_name,
                file=file_path,
                scale=[obj_scale, obj_scale, obj_scale],
            )
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=obj_density,
                contype=3,  # bits 1|2: collide with both the hand (bit 1) and the support floor (bit 2)
                conaffinity=3,
            )
        return

    def _set_friction(self, test_friction: float):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = test_friction
            g.condim = 4
        return

    def _qpos_to_ctrl(self, hand_qpos: np.ndarray) -> np.ndarray:
        if self.hand_mocap:
            return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]
        else:
            return self._qpos2ctrl_matrix @ hand_qpos

    def get_obj_pose(self) -> np.ndarray:
        return self.data.qpos[-7:]

    def get_contact_info(self, hand_qpos: np.ndarray, obj_pose: np.ndarray, obj_margin: float = 0) -> tuple:
        # Set margin and gap to detect contact
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = obj_margin

        # Set pose and qpos for hand and object
        self.reset_pose_qpos(hand_qpos, obj_pose)
        ho_contact, hh_contact = self.scan_contacts()

        # Set margin and gap back
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = 0
        return ho_contact, hh_contact

    def scan_contacts(self) -> tuple:
        """Read hand-object / hand-hand contacts from the CURRENT sim state (no reset)."""
        object_id = self.model.nbody - 1
        last_hand_id = object_id - (2 if self.hand_mocap else 1)

        # Processing all contact information
        ho_contact = []
        hh_contact = []
        for contact in self.data.contact:
            body1_id = int(self.model.geom_bodyid[contact.geom1])
            body2_id = int(self.model.geom_bodyid[contact.geom2])
            body1_name = self.model.body(body1_id).name
            body2_name = self.model.body(body2_id).name
            # hand and object
            if (1 <= body1_id <= last_hand_id and body2_id == object_id) or (
                1 <= body2_id <= last_hand_id and body1_id == object_id
            ):
                # keep body1=hand and body2=object
                if body2_id == object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact_normal,
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            # hand and hand
            elif 1 <= body1_id <= last_hand_id and 1 <= body2_id <= last_hand_id:
                hh_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact.frame[0:3],
                        "body1_name": body1_name,
                        "body2_name": body2_name,
                    }
                )
        return ho_contact, hh_contact

    def set_ext_force_on_obj(self, ext_force: np.ndarray):
        self.data.xfrc_applied[-1] = ext_force
        return

    def reset_pose_qpos(self, hand_qpos: np.ndarray, obj_pose: np.ndarray):
        # set key frame
        self.model.key_qpos[0] = np.concatenate([hand_qpos, obj_pose], axis=0)
        self.model.key_ctrl[0] = self._qpos_to_ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        if self.hand_mocap:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return

    def control_hand_with_interp(
        self,
        hand_qpos1: np.ndarray,
        hand_qpos2: np.ndarray,
        step_outer: int = 10,
        step_inner: int = 10,
    ):
        if self.hand_mocap:
            pose_interp = interpolate_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = interpolate_qpos(self._qpos_to_ctrl(hand_qpos1), self._qpos_to_ctrl(hand_qpos2), step_outer)
        for j in range(step_outer):
            if self.hand_mocap:
                self.data.mocap_pos[0] = pose_interp[j, :3]
                self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner)
        return

    def control_hand_step(self, step_inner: int):
        self.sim_steps += step_inner
        for _ in range(step_inner):
            mujoco.mj_step(self.model, self.data)
        if self.record:
            self.hand_qpos_log.append(self.data.qpos[:-7].copy())
            self.obj_pose_log.append(self.data.qpos[-7:].copy())
        return
