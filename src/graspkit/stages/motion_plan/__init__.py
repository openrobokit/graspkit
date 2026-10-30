from torch.types import Device

from graspkit.configs.experiment_config import MotionPlannerConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.stages.motion_plan.torch_planner import TorchPlanner
from graspkit.stages.motion_plan.warp_planner import WarpPlanner


def build_motion_planner(
    cfg: MotionPlannerConfig, robot_config: RobotConfig, device: Device
) -> "TorchPlanner | WarpPlanner":
    if cfg.backend == "torch":
        return TorchPlanner(cfg, robot_config=robot_config, device=device)
    if cfg.backend == "warp":
        return WarpPlanner(cfg, robot_config=robot_config, device=device)
    raise ValueError(f"Unknown motion planner backend: {cfg.backend!r}")


__all__ = [
    "TorchPlanner",
    "WarpPlanner",
    "build_motion_planner",
]
