from typing import TYPE_CHECKING

from torch.types import Device

from graspkit.configs.experiment_config import EnvConfig, RendererConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.envs.base import BaseEnv

if TYPE_CHECKING:
    from graspkit.mani_skill.renderer import MsRenderer

__all__ = ["BaseEnv", "MsEnv", "MjEnv", "MjNativeEnv", "IsaacEnv", "RealEnv", "build_env", "build_renderer"]


def build_env(
    cfg: EnvConfig | None,
    robot_config: RobotConfig,
    device: Device,
    enable_rendering: bool = True,
    renderer_cfg: RendererConfig | None = None,
) -> BaseEnv | None:
    """Build the selected simulator without importing unrelated backends; `renderer_cfg` configures `observe`."""
    if cfg is None:
        return None
    backends = (cfg.isaac_env_config, cfg.mj_env_config, cfg.mj_native_env_config)
    assert sum(backend is not None for backend in backends) <= 1, "Select one simulator backend."
    if cfg.isaac_env_config is not None:
        from graspkit.envs.isaac import IsaacEnv

        return IsaacEnv(cfg, robot_config, device, renderer_cfg)
    if cfg.mj_native_env_config is not None:
        from graspkit.envs.mujoco_native import MjNativeEnv

        return MjNativeEnv(cfg, robot_config, device, renderer_cfg)
    if cfg.mj_env_config is not None:
        from graspkit.configs.mj_hand import SHADOW_HAND, VEGA_SHARPA
        from graspkit.envs.mujoco import MjEnv

        floating = robot_config.base_type == "floating"
        hand = cfg.mj_env_config.hand or (SHADOW_HAND if floating else VEGA_SHARPA)
        base_pos = None if floating else robot_config.init_position
        base_quat = None if floating else robot_config.init_quaternion
        return MjEnv(
            cfg.mj_env_config,
            robot_config=robot_config,
            hand=hand,
            num_workers=cfg.mj_env_config.num_workers,
            base_pos=base_pos,
            base_quat=base_quat,
        )
    from graspkit.envs.maniskill import MsEnv

    return MsEnv(
        cfg, robot_config=robot_config, device=device, renderer_cfg=renderer_cfg, enable_rendering=enable_rendering
    )


def build_renderer(cfg: RendererConfig, robot_config: RobotConfig, device: Device) -> "MsRenderer":
    """Build the training-data renderer; only ManiSkill renders, so `cfg.env_config` must select it."""
    backends = (cfg.env_config.isaac_env_config, cfg.env_config.mj_env_config, cfg.env_config.mj_native_env_config)
    assert all(backend is None for backend in backends), "Only ManiSkill renders."
    from graspkit.mani_skill.renderer import MsRenderer

    return MsRenderer(cfg, robot_config, device)


def __getattr__(name: str):
    if name == "MsEnv":
        from graspkit.envs.maniskill import MsEnv

        return MsEnv
    if name == "MjEnv":
        from graspkit.envs.mujoco import MjEnv

        return MjEnv
    if name == "MjNativeEnv":
        from graspkit.envs.mujoco_native import MjNativeEnv

        return MjNativeEnv
    # RealEnv pulls hardware deps (retargetingkit/Realsense); import it lazily so the sim path never requires `xarm`.
    if name == "RealEnv":
        from graspkit.envs.real import RealEnv

        return RealEnv
    if name == "IsaacEnv":
        from graspkit.envs.isaac import IsaacEnv

        return IsaacEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
