from typing import Literal, Union

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Float
from robokit.robo.robot import Robot


def _maniskill_joint_names(env: gym.Env, space: Literal["observation", "action"]) -> list:
    """ManiSkill exposes two DIFFERENT joint orders that diverge once any DOF sits in a named sub-controller:
    - "observation": `agent.get_proprioception()`'s qpos is `agent.robot.get_qpos()`, the native
      per-articulation joint order.
    - "action": `CombinedController.set_action` slices the flat action per named sub-controller,
      in `agent.controller.joints` order (each sub-controller's own joint list, concatenated).
    Picking the wrong space can reindex to the wrong joint (non-contiguous DOF grouping broke vega_sharpa_right).
    """
    if space == "observation":
        return [joint.name for joint in env.unwrapped.agent.robot.active_joints]
    return [joint.name for joint in env.unwrapped.agent.controller.joints]


@torch.no_grad()
def reindex_qpos_maniskill_to_ours(
    qpos: Union[Float[torch.Tensor, "... num_dofs"], Float[np.ndarray, "... num_dofs"]],
    robot: Robot,
    env: gym.Env,
    space: Literal["observation", "action"] = "observation",
) -> Union[Float[torch.Tensor, "... num_dofs"], Float[np.ndarray, "... num_dofs"]]:
    """`qpos` is ManiSkill-ordered per `space` (default "observation", e.g. `obs["agent"]["qpos"]`; or "action")."""
    ms_joint_names = _maniskill_joint_names(env, space)
    floating_base = len(robot.spec.nonfixed_joint_names) == len(ms_joint_names) - 6

    reindex = [ms_joint_names.index(name) for name in robot.actuated_joint_names]

    if isinstance(qpos, torch.Tensor):
        reindexed = qpos[..., reindex]
        if floating_base:
            return torch.cat([qpos[..., :6], reindexed], dim=-1).clone().contiguous()
        else:
            return reindexed.clone().contiguous()
    else:
        reindexed = qpos[..., reindex]
        if floating_base:
            return np.concatenate([qpos[..., :6], reindexed], axis=-1).copy()
        else:
            return reindexed.copy()


@torch.no_grad()
def reindex_qpos_ours_to_maniskill(
    qpos: Union[Float[torch.Tensor, "... num_dofs"], Float[np.ndarray, "... num_dofs"]],
    robot: Robot,
    env: gym.Env,
    space: Literal["observation", "action"] = "action",
) -> Union[Float[torch.Tensor, "... num_dofs"], Float[np.ndarray, "... num_dofs"]]:
    """Returns a ManiSkill-ordered array per `space` (default "action" for `env.step`; or "observation")."""
    ours_full_joint_names = robot.spec.joint_names
    ms_joint_names = _maniskill_joint_names(env, space)
    floating_base = len(robot.spec.nonfixed_joint_names) == len(ms_joint_names) - 6

    if floating_base:
        reindex = [ours_full_joint_names.index(name) for name in ms_joint_names[6:]]
        free_qpos, non_free_qpos = qpos[..., :6], qpos[..., 6:]
    else:
        reindex = [ours_full_joint_names.index(name) for name in ms_joint_names]
        non_free_qpos = qpos

    if isinstance(qpos, torch.Tensor):
        full_qpos = robot.map_to_full_joint_values_torch(non_free_qpos)
        reindexed = full_qpos[..., reindex]
        if floating_base:
            return torch.cat([free_qpos, reindexed], dim=-1).clone().contiguous()
        else:
            return reindexed.clone().contiguous()
    else:
        non_free_tensor = torch.from_numpy(non_free_qpos.astype(np.float32))
        full_qpos = robot.map_to_full_joint_values_torch(non_free_tensor).numpy()
        reindexed = full_qpos[..., reindex]
        if floating_base:
            return np.concatenate([free_qpos, reindexed], axis=-1).copy()
        else:
            return reindexed.copy()
