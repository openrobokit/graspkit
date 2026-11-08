"""Robokit upper-body IK: track one or both hand frames, solve torso (height + waist 3-DOF) + arms.

Lifted from humanoid_locomanip/hal/solver/robokit_upper_ik.py, with the robokit surface it needs
(robot config, RobotHelper, IK-config builders) distilled here. Output q21 layout is
``[torso(4), head(3), left_arm(7), right_arm(7)]``; the head slots are frozen by the mask, never solved.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin
import warp as wp
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.robo import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from scipy.spatial.transform import Rotation as R

from hug import paths
from hug.utils import LPFilter, SE3Filter


@dataclass(frozen=True)
class G1InspireUpperConfig:
    """Robot config for the G1 + Inspire upper-body URDF (33 actuated joints)."""

    urdf_or_mjcf_path: str = field(default_factory=lambda: str(paths.G1_INSPIRE_UPPER_HEAD_URDF))
    left_arm_base_link_name: str = "torso_link"
    right_arm_base_link_name: str = "torso_link"
    left_hand_base_link_name: str = "L_hand_base_link"
    right_hand_base_link_name: str = "R_hand_base_link"


@lru_cache(maxsize=1)
def g1_inspire_upper() -> G1InspireUpperConfig:
    """The shared upper-body config, built on first use so importing resolves no asset."""
    return G1InspireUpperConfig()


# Task names pinned via IKSolverConfig.add(name=...), so load_config can re-weight per link.
POSITION_TASK_NAME = "position"
ROTATION_TASK_NAME = "rotation"


class RobotHelper:
    """Upper-body IK helper: the full robot plus per-arm sub-robots, used only for the arm joint indices."""

    def __init__(
        self,
        robot_config: Optional[G1InspireUpperConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.robot_config = robot_config if robot_config is not None else g1_inspire_upper()
        self.device = device

        self.robot = Robot.load(
            robot_description_or_path=self.robot_config.urdf_or_mjcf_path,
            load_meshes=False,
        )
        left_arm_robot = Robot.load(
            robot_description_or_path=self.robot_config.urdf_or_mjcf_path,
            load_meshes=False,
            base_link_name=self.robot_config.left_arm_base_link_name,
            ee_link_names=[self.robot_config.left_hand_base_link_name],
        )
        right_arm_robot = Robot.load(
            robot_description_or_path=self.robot_config.urdf_or_mjcf_path,
            load_meshes=False,
            base_link_name=self.robot_config.right_arm_base_link_name,
            ee_link_names=[self.robot_config.right_hand_base_link_name],
        )

        full_names = self.robot.spec.actuated_joint_names
        self.left_arm_joint_indices = np.array(
            [full_names.index(jn) for jn in left_arm_robot.spec.actuated_joint_names],
            dtype=np.int64,
        )
        self.right_arm_joint_indices = np.array(
            [full_names.index(jn) for jn in right_arm_robot.spec.actuated_joint_names],
            dtype=np.int64,
        )

    def create_multi_frame_ik_helper(
        self,
        target_frames: list[str],
        batch_size: int = 1,
        ik_config: Optional[IKSolverConfig] = None,
    ) -> IKSolver:
        """An :class:`IKSolver` over ``target_frames``, warmed up at ``batch_size``."""
        if ik_config is None:
            ik_config = create_realtime_ik_config(self)
        wp_device = "cuda:0" if self.device != "cpu" else "cpu"
        helper = IKSolver(ik_config, self.robot, target_frames, wp_device)
        helper.warmup(batch_size)
        return helper


def apply_joint_limits(helper: RobotHelper, config: "RobokitWBCConfig") -> None:
    """Write ``config``'s torso and arm limits into ``helper.robot.spec`` (torso 0:4, arms by the helper's indices)."""
    spec = helper.robot.spec
    spec.actuated_joint_limits[0:4, 0] = config.torso_lower_limits
    spec.actuated_joint_limits[0:4, 1] = config.torso_upper_limits
    spec.actuated_joint_limits[helper.left_arm_joint_indices, 0] = config.left_arm_lower_limits
    spec.actuated_joint_limits[helper.left_arm_joint_indices, 1] = config.left_arm_upper_limits
    spec.actuated_joint_limits[helper.right_arm_joint_indices, 0] = config.right_arm_lower_limits
    spec.actuated_joint_limits[helper.right_arm_joint_indices, 1] = config.right_arm_upper_limits


def create_realtime_ik_config(
    helper: RobotHelper, dt: float = 0.01, config: Optional["RobokitWBCConfig"] = None
) -> IKSolverConfig:
    """Real-time :class:`IKSolverConfig` for the G1 upper body.

    Side effect: writes the joint limits of ``config`` (default ``RobokitWBCConfig()``) and the velocity limits."""
    num_joints = helper.robot.num_actuated_joints
    apply_joint_limits(helper, config if config is not None else RobokitWBCConfig())

    helper.robot.spec.actuated_joint_velocity_limits[0] = 0.08
    helper.robot.spec.actuated_joint_velocity_limits[1] = 0.16

    smoothness_weights = np.full(num_joints, 1.0, dtype=np.float32)
    smoothness_weights[0] = 7.5
    smoothness_weights[1] = 1.5

    return (
        IKSolverConfig(enable_T_world_base=False)
        .add(
            PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=3.0),
            name=POSITION_TASK_NAME,
        )
        .add(
            RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.5),
            name=ROTATION_TASK_NAME,
        )
        .add(PositionLimit(weight=50.0))
        .add(ConfigurationTask(rest_q=IKContext.REST_Q, weight=0.01))
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=smoothness_weights))
        .add(VelocityLimitTask(dt=dt, prev_state_var=IKContext.PREVIOUS_STATE, weight=1.0))
    )


@dataclass
class RobokitWBCConfig:
    device: str = "cuda"
    track_hands: str = "right"  # "both" | "left" | "right"
    hand_position_weight: float = 3.0  # IK task weights for the tracked hands
    hand_rotation_weight: float = 1.5
    arm_in_lp_alpha: float = 0.95  # filter alphas
    waist_out_lp_alpha: float = 0.99
    arm_out_lp_alpha: float = 0.95
    # joint limits written into the robokit spec by create_realtime_ik_config
    torso_lower_limits: list[float] = field(default_factory=lambda: [0.25, -0.25, 0.0, 0.0])
    torso_upper_limits: list[float] = field(default_factory=lambda: [0.75, 0.25, 0.0, 0.0])
    left_arm_lower_limits: list[float] = field(
        default_factory=lambda: [-1.57, 0.0, -1.57, -1.0472, -1.972, -1.50, -1.50]
    )
    left_arm_upper_limits: list[float] = field(default_factory=lambda: [0.78, 1.55, 1.57, 1.35, 1.50, 1.50, 1.50])
    right_arm_lower_limits: list[float] = field(
        default_factory=lambda: [-1.57, -1.55, -1.57, -1.0472, -1.972, -1.50, -1.50]
    )
    right_arm_upper_limits: list[float] = field(default_factory=lambda: [0.78, 0.0, 1.57, 1.35, 1.50, 1.50, 1.50])

    # rest pose of an un-tracked arm, frozen there by the active-joint mask
    default_left_arm: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.785, 1.34, 0.0, 0.0, 0.0])
    default_right_arm: list[float] = field(default_factory=lambda: [0.0, -0.1, -0.785, 1.34, 0.0, 0.0, 0.0])

    @property
    def default_qpos(self) -> np.ndarray:
        # fmt: off
        return np.array([
            0.75, 0.0, 0.0, 0.0,                 # torso: height, waist yaw, waist roll, waist pitch
            0.0, 0.0, 0.0,                       # head: yaw, roll, pitch (frozen)
            0.0, 0.3, 0.0, -0.5, 0.0, 0.0, 0.0,  # left arm
            0.0, -0.3, 0.0, -0.5, 0.0, 0.0, 0.0, # right arm
        ])
        # fmt: on


class RobokitWBController:
    """Upper-body IK over ``[right_wrist, left_wrist]``; the active mask freezes the head and any un-tracked arm."""

    def __init__(self, config: Optional[RobokitWBCConfig] = None, dt: float = 1 / 50):
        self.dt = float(dt)
        self.config = config if config is not None else RobokitWBCConfig()
        self.configured = False
        self._qpos: Optional[np.ndarray] = None
        self.dof = 21

    def load_config(self) -> None:
        wp.config.log_level = wp.LOG_WARNING  # no init banner, no per-kernel load lines
        if not (Path.home() / ".cache" / "warp" / wp.config.version).exists():  # warp's default kernel cache
            print("[wbc] first run: compiling the IK's GPU kernels (about 2 min, cached afterwards) ...")
        wp.init()

        self.device = self.config.device if wp.is_cuda_available() else "cpu"
        self.wp_device = "cuda:0" if self.device != "cpu" else "cpu"

        self.robot_helper = RobotHelper(g1_inspire_upper(), device=self.device)

        urdf_path = self.robot_helper.robot_config.urdf_or_mjcf_path
        self.pin_model = pin.buildModelFromUrdf(str(urdf_path))
        self.pin_data = self.pin_model.createData()

        right_frame = self.robot_helper.robot_config.right_hand_base_link_name
        left_frame = self.robot_helper.robot_config.left_hand_base_link_name
        target_frames = [right_frame, left_frame]

        mode = self.config.track_hands
        self.track_left = mode in ("both", "left")
        self.track_right = mode in ("both", "right")
        if not (self.track_left or self.track_right):
            raise ValueError(f"Invalid track_hands: {mode!r}. Must be 'both', 'left', or 'right'")

        ik_config = create_realtime_ik_config(self.robot_helper, dt=self.dt, config=self.config)

        self.ik_helper = self.robot_helper.create_multi_frame_ik_helper(
            target_frames=target_frames,
            batch_size=1,
            ik_config=ik_config,
        )

        # Joint-group indices into the full 33-DOF robokit qpos.
        self.left_arm_idx = self.robot_helper.left_arm_joint_indices
        self.right_arm_idx = self.robot_helper.right_arm_joint_indices
        spec = self.robot_helper.robot.spec
        joint_names = spec.actuated_joint_names
        self.torso_idx = np.array(
            [i for i, n in enumerate(joint_names) if "waist" in n or "height_base" in n],
            dtype=np.int64,
        )
        self.head_idx = np.array([i for i, n in enumerate(joint_names) if n.startswith("head_")], dtype=np.int64)
        assert self.torso_idx.tolist() == [0, 1, 2, 3]
        assert self.head_idx.tolist() == [4, 5, 6]

        # Active mask: torso and tracked arms on, head off.
        num_joints = spec.num_actuated_joints
        active_mask = np.zeros(num_joints, dtype=np.float32)
        active_mask[self.torso_idx] = 1.0
        if self.track_left:
            active_mask[self.left_arm_idx] = 1.0
        if self.track_right:
            active_mask[self.right_arm_idx] = 1.0

        ap, ar = self.config.hand_position_weight, self.config.hand_rotation_weight
        pos_weights = [ap if self.track_right else 0.0, ap if self.track_left else 0.0]
        rot_weights = [ar if self.track_right else 0.0, ar if self.track_left else 0.0]

        # Weights are ordered ``[right_frame, left_frame]``, matching the IK construction above.
        self.ik_helper.set_weight(POSITION_TASK_NAME, pos_weights)
        self.ik_helper.set_weight(ROTATION_TASK_NAME, rot_weights)
        self.ik_helper.set_active_joint_mask(active_mask.tolist())

        self.left_wrist_filter = SE3Filter(self.config.arm_in_lp_alpha)
        self.right_wrist_filter = SE3Filter(self.config.arm_in_lp_alpha)
        self.torso_filter = LPFilter(self.config.waist_out_lp_alpha)
        self.left_arm_filter = LPFilter(self.config.arm_out_lp_alpha)
        self.right_arm_filter = LPFilter(self.config.arm_out_lp_alpha)

        self._qpos = self._with_default_untracked_arms(self.config.default_qpos)
        self.prev_state = self._make_state(self._qpos)

        self.configured = True

    def _with_default_untracked_arms(self, q21: np.ndarray) -> np.ndarray:
        q = np.asarray(q21, dtype=np.float32).copy()
        if not self.track_left:
            q[7:14] = self.config.default_left_arm
        if not self.track_right:
            q[14:21] = self.config.default_right_arm
        return q

    def _expand_q21_to_full(self, q21: np.ndarray) -> np.ndarray:
        full = self.robot_helper.robot.spec.midrange_q.copy().astype(np.float32)
        full[self.torso_idx] = q21[0:4]
        full[self.head_idx] = q21[4:7]
        full[self.left_arm_idx] = q21[7:14]
        full[self.right_arm_idx] = q21[14:21]
        return full

    def _make_state(self, q21: np.ndarray):
        full = self._expand_q21_to_full(q21)
        return self.robot_helper.robot.state(
            q=wp.from_numpy(full.reshape(1, -1), dtype=wp.float32, device=self.wp_device)
        )

    def _to_pose7(self, mat: np.ndarray) -> np.ndarray:
        position = mat[:3, 3]
        quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        return np.concatenate([position, quat_wxyz]).astype(np.float32)

    def reset(self, q: np.ndarray) -> None:
        """Reset controller state to provided 21-DOF pose [torso(4), head(3), left(7), right(7)]."""
        if not self.configured:
            raise ValueError("RobokitWBController has not been configured.")

        q = self._with_default_untracked_arms(q)
        self.prev_state = self._make_state(q)
        self._qpos = q.copy()

        self.left_wrist_filter.reset()
        self.right_wrist_filter.reset()
        self.torso_filter.reset()
        self.left_arm_filter.reset()
        self.right_arm_filter.reset()

    def update(
        self,
        right_wrist2world_mat: Optional[np.ndarray] = None,
        left_wrist2world_mat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve one step; returns q21 ``[torso(4), head(3), left(7), right(7)]``. ``None`` wrist = un-tracked."""
        if not self.configured:
            raise ValueError("RobokitWBController has not been configured.")

        if right_wrist2world_mat is None:
            right_wrist2world_mat = np.eye(4)
        if left_wrist2world_mat is None:
            left_wrist2world_mat = np.eye(4)

        right_mat = np.array(self.right_wrist_filter.update(pin.SE3(right_wrist2world_mat)))
        left_mat = np.array(self.left_wrist_filter.update(pin.SE3(left_wrist2world_mat)))

        targets = np.stack([self._to_pose7(right_mat), self._to_pose7(left_mat)])[None]  # (1, links, 7)
        solved_state = self.ik_helper.solve_numpy(targets, prev_state=self.prev_state, init_state=self.prev_state)
        self.prev_state = solved_state

        solved_q = solved_state.q.numpy()[0]
        torso_qpos = self.torso_filter.next(solved_q[self.torso_idx])
        left_arm_qpos = self.left_arm_filter.next(solved_q[self.left_arm_idx])
        right_arm_qpos = self.right_arm_filter.next(solved_q[self.right_arm_idx])
        head_qpos = solved_q[self.head_idx]  # frozen by mask

        self._qpos[0:4] = torso_qpos
        self._qpos[4:7] = head_qpos
        self._qpos[7:14] = left_arm_qpos
        self._qpos[14:21] = right_arm_qpos

        return np.concatenate([torso_qpos, head_qpos, left_arm_qpos, right_arm_qpos])

    def get_pelvis_frame(self) -> pin.SE3:
        """The IK's assumed pelvis: identity rotation at ``[0, 0, height]``, the root before the waist 3-DOF."""
        height = float(self.prev_state.q.numpy()[0, 0])
        return pin.SE3(np.eye(3), np.array([0.0, 0.0, height]))

    @property
    def qpos(self) -> np.ndarray:
        return self._qpos.astype(np.float32).copy()

    def solve_gravity_compensation(
        self,
        q: np.ndarray,
        waist_yrp: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> np.ndarray:
        """Unclipped gravity-compensation torques: 14 arm joints (left + right), then waist yaw/roll/pitch.

        ``gravity`` is expressed in the pelvis frame (the model's root), so a tilted pelvis is a rotated vector."""
        full_q = np.zeros(self.pin_model.nq)
        full_qvel = np.zeros(self.pin_model.nv)
        full_qacc = np.zeros(self.pin_model.nv)
        full_q[0] = 0.75
        full_q[1:4] = waist_yrp
        full_q[7:14] = q[0:7]
        full_q[26:33] = q[7:14]
        self.pin_model.gravity.linear = np.asarray(gravity, dtype=np.float64)
        tau = pin.rnea(self.pin_model, self.pin_data, full_q, full_qvel, full_qacc)
        return np.concatenate([tau[7:14], tau[26:33], tau[1:4]])
