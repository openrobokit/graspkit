"""Reflect a floating-base PointNet checkpoint across Y, e.g. a right-hand policy into native left coordinates."""

import graspkit  # noqa: F401 I001

from pathlib import Path

import torch
import tyro
from torch import nn

from graspkit.configs.experiment import build_experiment_config
from graspkit.models.grasp_policy import GraspPolicy, build_grasp_policy


@torch.no_grad()
def mirror_floating_pointnet_model(policy: GraspPolicy):
    """Reflect a policy across Y in place (mirrored hand joint conventions must match); no mirror wrapper is needed."""
    assert policy.cfg.vision_encoder == "pointnet"
    assert policy.cfg.grasp_evaluator_guidance_steps == 0, "Mirror the base policy before adding a grasp evaluator"
    assert policy.robot_config.base_type == "floating" and policy.cfg.rot_begin_idx == 3
    rotation_signs = [1, -1, 1, -1, 1, -1] if policy.cfg.convert_rxyz_to_rot6d else [-1, 1, -1]
    signs = policy.vision_encoder.conv1.weight.new_ones(policy.cfg.num_queries, policy.action_dim)
    signs[:, 1] = -1
    signs[:, 3 : 3 + len(rotation_signs)] = signs.new_tensor(rotation_signs)
    signs = signs.flatten()
    policy.vision_encoder.conv1.weight[:, 1].neg_()
    encoder, decoder = policy.generator.encoder, policy.generator.decoder
    assert isinstance(encoder.skip_output, nn.Linear) and isinstance(decoder.skip_output, nn.Linear)
    for layer in (encoder.layers[0][0], encoder.skip_output):
        layer.weight[:, : len(signs)].mul_(signs)
    for layer in (decoder.layers[-1][0], decoder.skip_output):
        layer.weight.mul_(signs[:, None])
        layer.bias.mul_(signs)


def main(experiment: str, source: Path, output: Path):
    """`experiment` is the mirrored robot's experiment, e.g. `sharpa_left_dex1b_v0_5_mini_tabletop_gd_zed`."""
    cfg = build_experiment_config(experiment)
    policy = build_grasp_policy(cfg.agent, cfg.robot_config, device="cpu")
    policy.load_state_dict(torch.load(source, map_location="cpu", weights_only=True)["agent"], strict=True)
    mirror_floating_pointnet_model(policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": policy.state_dict(),
            "experiment": cfg.name,
            "transfer": "y_reflection",
        },
        output,
    )
    print(f"Mirrored checkpoint: {output}")


if __name__ == "__main__":
    tyro.cli(main)
