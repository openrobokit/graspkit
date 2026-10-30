"""`RealEnvConfig`: hardware factories + rig geometry for `RealEnv`; instances live in `configs/real_env.py`."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class RealEnvConfig:
    name: str
    robot_config_name: str  # assembly RobotConfig key (ROBOT_CONFIGS) driving robot_helper/IK
    make_arm: Callable[[], object]  # () -> connected arm
    make_hand: Callable[[], object]  # () -> connected hand
    workspace_center: Tuple[float, float, float]  # policy point-cloud crop center, robot-base frame
    make_camera: Callable[[], object]  # () -> started camera
    hand_reindex: Optional[List[int]] = None  # ours (robot_helper order) -> hardware channel order
    # Deploy planning table (axis-aligned box scene): footprint of the real table this rig stands at.
    table_size: Tuple[float, float, float] = (1.209, 2.418, 0.9196429)
    table_xy: Tuple[float, float] = (-0.12, 0.0)  # its center in the world (scene) frame
    table_z: float = 0.0  # measured world-frame table-top height
    control_hz: float = 25.0  # step_qpos pacing; dexcontrol arms track setpoint streams best at >=50
    depth_frames: int = 1  # frames median-combined per capture; ZED per-frame depth is noisy
    pc_filter_radius: float = 0.0  # metres; zero disables radius outlier filtering
