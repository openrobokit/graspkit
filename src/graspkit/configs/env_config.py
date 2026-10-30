"""Configuration for ManiSkill, MuJoCo and the isolated Isaac runtime."""

from dataclasses import dataclass, field
from typing import Optional

from graspkit.constants import MUJOCO_EVAL_ROOT


@dataclass(frozen=True)
class CameraSpec:
    """A camera on a robot link or the world, set by a vertical fov in degrees or by a pinhole intrinsic."""

    name: str
    mount: Optional[str]
    """Link the camera is mounted on; None: the world."""
    pose: tuple[float, float, float, float, float, float, float]
    """[xyz, wxyz] in the mount frame."""
    fov: Optional[float] = None
    intrinsic: Optional[tuple[tuple[float, float, float], ...]] = None
    width: int = 320
    height: int = 240


@dataclass
class IsaacEnvConfig:
    """Isaac runtime settings; the task (control rate, object mass, lift height) is `EnvConfig`'s."""

    worker_command: list[str] = field(default_factory=lambda: ["/isaac-sim/python.sh"])
    """Isaac Python command, optionally prefixed by a foreground container launcher."""
    runtime_threads: int = 3
    """Kit tasking threads; the validated image stalls during reset with one Kit thread."""
    timeout: float = 900.0
    """Seconds to wait for each worker reply; the first reset imports the robot and compiles shaders."""
    max_envs: int = 64
    substeps: int = 2
    """Physics steps per control step: 100 Hz physics at the default 50 Hz control."""


@dataclass
class MjNativeEnvConfig:
    """Closed-loop MuJoCo settings (`graspkit.envs.mujoco_native`); the task itself is `EnvConfig`'s."""

    timestep: float = 0.002
    """Physics step; a control step at `EnvConfig.control_freq` runs a whole number of them."""
    solref: tuple[float, float] = (0.004, 1.0)
    """Contact time constant and damping ratio of the robot, table and object geoms (at least two timesteps)."""
    solimp: tuple[float, float, float, float, float] = (0.99, 0.999, 0.001, 0.5, 2.0)
    """Contact impedance; MuJoCo scales contact stiffness by the effective mass, so the 10 g object needs a stiff
    setting to hold a grip of tens of newtons without sinking in."""
    impratio: float = 10.0
    """Frictional-to-normal impedance ratio of the elliptic cones; higher resists slip in a grasp."""
    num_threads: int = 16
    """Threads stepping the scenes in parallel."""


@dataclass
class MsEnvConfig:
    """ManiSkill-only settings; the task itself is `EnvConfig`'s."""

    robot_uid: Optional[str] = None
    """ManiSkill agent uid; None registers one from the robot config. A hand-only uid on a floating-base robot means
    a static base."""
    reconfiguration_freq: Optional[int] = 1
    """Reconfiguration frequency."""


@dataclass
class HandSpec:
    xml_path: str
    mocap: bool
    finger_prefix: list
    valid_body_name: list
    tip_body_name: list
    exclude_table_contact: list[str] | None = None
    lift_qpos_index: int = 2
    """qpos index raised by LIFT_DIST for the tabletop lift: mocap root z (2), or a fixed-base arm's prismatic joint."""
    lift_arm_body: str = None
    """If set, the tabletop lift IKs the arm to raise this body +LIFT_DIST in z (fixed-base arm, no base joint)."""
    arm_qpos_indices: tuple = ()
    """qpos indices of the arm joints moved by the arm-IK lift (e.g. 0..6 for xarm, R_arm's slot for vega)."""


@dataclass
class MjEvalConfig:
    """MuJoCo grasp-evaluation settings."""

    num_workers: int = 48
    obj_mass: float = 0.1
    """Object mass (kg)."""
    miu_coef: tuple = (0.6, 0.02)
    """[tangential, torsional] friction."""
    setting: str = "fc"
    """Evaluation setting; only `"fc"` is ported."""
    max_pene: float = 0.01
    trans_thre: float = 0.05
    lift_slip_thre: float = 0.02
    """Tabletop: max object-vs-palm slip during the lift (tight; trans_thre is too loose)."""
    min_lift_dist: float = 0.0
    """Tabletop: minimum actual object rise; zero preserves existing evaluator behavior."""
    grasp_interp_steps: int = 25
    """Interp frames for pregrasp->grasp and grasp->squeeze."""
    lift_interp_steps: int = 50
    """Interp frames for the squeeze->lift raise."""
    angle_thre: float = 15.0
    contact_margin: float = 0.01
    contact_threshold: float = 0.002
    fc_force_scale: float = 10.0
    """Multiplies obj_mass for the fc/tabletop external force (lower = gentler test)."""
    timestep: float = 0.004
    """MuJoCo step; lower (e.g. 0.001) stabilizes an actuated arm+hand grip."""
    solref_timeconst: float = 0.005
    """Rigid-contact solref[0]; smaller = more rigid (less penetration)."""
    obj_root: str = str(MUJOCO_EVAL_ROOT / "example_object")
    hand: HandSpec = None
    """None -> build_env picks the default (floating SHADOW_HAND, or fixed-base VEGA_SHARPA)."""
