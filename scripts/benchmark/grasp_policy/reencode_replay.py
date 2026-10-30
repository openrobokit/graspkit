"""Re-encode a replay's features with another actor's encoder so `train_grasp_evaluator.py` can reuse the labels.

    uv run scripts/benchmark/grasp_policy/reencode_replay.py --checkpoint ACTOR.pt --replay replay4096.pt \
        --observations observations4096.h5 --scenes scenes.msgpack --output replay_actor.pt
"""

import graspkit  # noqa: F401 I001

from dataclasses import replace
from pathlib import Path

import torch
import tyro

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.datasets.grasp_dataset import GraspDataset
from graspkit.models.grasp_policy import build_grasp_policy
from graspkit.utils.seed_utils import seed_everything


def main(checkpoint: Path, replay: Path, observations: Path, scenes: Path, output: Path, batch: int = 64):
    seed_everything(0)
    assert not output.exists()
    saved = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
    cfg = build_experiment_config(saved["experiment"])
    policy_cfg = GraspPolicyConfig(**saved["policy_config"])
    assert policy_cfg.vision_encoder_global_feature
    policy = build_grasp_policy(policy_cfg, cfg.robot_config, "cuda:0").eval().requires_grad_(False)
    policy.load_state_dict(saved["agent"], strict=True)
    data = torch.load(replay, map_location="cpu", weights_only=True)
    dataset = GraspDataset(
        replace(
            cfg.dataset,
            return_actions=False,
            return_graspness=False,
            return_rgb=policy_cfg.vision_encoder.startswith("rgb"),
        ),
        str(observations),
        str(scenes),
        split="train",
    )
    lookup = {sid: i for i, sid in enumerate(dataset.scene_ids)}
    ids = data["scene_ids"]
    assert set(ids) <= set(lookup) and data["features"].shape[1] == 1
    features = []
    captured = []
    hook = (
        policy.generator.cond_proj if policy_cfg.generative_model == "flow_transformer" else policy.generator.decoder
    ).register_forward_pre_hook(
        lambda module, inputs: captured.append(
            inputs[0]
            if policy_cfg.generative_model == "flow_transformer"
            else inputs[0][
                :, policy.generator.latent_dim : policy.generator.latent_dim + policy_cfg.vision_encoder_feature_dim
            ]
        )
    )
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            samples = [dataset[lookup[sid]]["observations"] for sid in ids[start : start + batch]]
            obs = {key: value.cuda() for key, value in torch.utils.data.default_collate(samples).items()}
            captured.clear()
            policy.get_action(obs)
            assert captured and all(torch.equal(value, captured[0]) for value in captured)
            features.append(captured[0].cpu())
            print(f"encoded {min(start + batch, len(ids))}/{len(ids)}", flush=True)
    hook.remove()
    features = torch.cat(features)[:, None]
    assert features.shape == data["features"].shape and torch.isfinite(features).all()
    dataset.h5_file.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **data,
            "features": features,
        },
        output,
    )
    print(f"{len(ids)} observations re-encoded -> {output}")


if __name__ == "__main__":
    tyro.cli(main)
