import graspkit  # noqa: F401 I001

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
import tyro
from torch.nn import functional as F

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.datasets.grasp_dataset import GraspDataset
from graspkit.models.grasp_evaluator import GraspEvaluator
from graspkit.models.grasp_policy import build_grasp_policy
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.seed_utils import seed_everything


def main(
    experiment: str,
    replay: Path,
    checkpoint: Path,
    output: Path,
    seed: int = 0,
    iterations: int = 10080,
    validation_scenes: int = 128,
    guidance_steps: int = 10,
    guidance_radius: float = 2.0,
    ranking_weight: float = 0.0,
    observations: Path | None = None,
    local_features: bool = False,
    anchor_state: bool = False,
):
    """Train a success scorer on cached rollouts and package a guided policy; no test scenes, so validate separately."""
    assert not output.exists() and iterations > 0 and guidance_steps > 0 and guidance_radius > 0
    assert ranking_weight >= 0
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    cfg = build_experiment_config(experiment)
    source = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
    cfg = replace(
        cfg,
        agent=GraspPolicyConfig(**source["policy_config"]),
        scene_description_path=source["scene_description_path"],
        processed_data_path=source["processed_data_path"],
    )
    saved = torch.load(replay, map_location="cpu", weights_only=True)
    ids = saved["scene_ids"]
    splits = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path, return_splits=True)
    assert len(ids) == len(set(ids)) > validation_scenes > 0
    assert set(ids) <= set(splits["train"]) and not set(ids) & set(splits["test"])
    assert not set(ids) & set(saved["reserved_validation_ids"])
    features, actions, rewards = saved["features"].cuda(), saved["actions"].cuda(), saved["rewards"].cuda()
    assert actions.shape[:2] == rewards.shape and features.shape[0] == len(ids) == len(actions)
    assert torch.isfinite(features).all() and torch.isfinite(actions).all()
    assert ((rewards >= 0) & (rewards <= 1)).all()
    assert rewards.any() and (rewards == 0).any()
    assert not cfg.agent.grasp_evaluator_local_features or local_features
    assert not cfg.agent.grasp_evaluator_anchor_state or anchor_state
    assert not local_features or (
        observations is not None and ranking_weight > 0 and cfg.agent.grasp_evaluator_guidance_steps
    )
    assert not anchor_state or local_features
    policy = (
        build_grasp_policy(cfg.agent, cfg.robot_config, "cuda:0")
        .eval()
        .requires_grad_(observations is not None and not local_features)
    )
    policy.load_state_dict(source["agent"], strict=True)
    assert features.shape[-1] == cfg.agent.vision_encoder_feature_dim
    assert actions.shape[-1] == policy.action_dim * cfg.agent.num_queries
    features = features.reshape(len(ids), -1, features.shape[-1])
    assert actions.shape[1] % features.shape[1] == 0
    features = features.unsqueeze(2).expand(-1, -1, actions.shape[1] // features.shape[1], -1).flatten(1, 2)
    if ranking_weight:
        assert actions.shape[1] % 64 == 0
        panels = features.reshape(len(ids), -1, 64, features.shape[-1])
        assert torch.equal(panels, panels[:, :, :1].expand_as(panels))
    predictor = (
        policy.grasp_evaluator.requires_grad_(True)
        if cfg.agent.grasp_evaluator_guidance_steps
        else GraspEvaluator(
            features.shape[-1],
            cfg.robot_config,
            cfg.agent.num_queries,
            hand_joint_features=cfg.agent.grasp_evaluator_hand_joint_features,
            mask_constant_features=cfg.agent.grasp_evaluator_mask_constant_features,
        ).cuda()
    )
    if local_features and not cfg.agent.grasp_evaluator_local_features:
        predictor = GraspEvaluator(
            features.shape[-1],
            cfg.robot_config,
            cfg.agent.num_queries,
            True,
            hand_joint_features=cfg.agent.grasp_evaluator_hand_joint_features,
            mask_constant_features=cfg.agent.grasp_evaluator_mask_constant_features,
        ).cuda()
        predictor.network.load_state_dict(policy.grasp_evaluator.network.state_dict(), strict=True)
        predictor.center.copy_(policy.grasp_evaluator.center)
        predictor.deviation.copy_(policy.grasp_evaluator.deviation)
    if anchor_state and not cfg.agent.grasp_evaluator_anchor_state:
        original = predictor.state_dict()
        predictor = GraspEvaluator(
            features.shape[-1],
            cfg.robot_config,
            cfg.agent.num_queries,
            True,
            True,
            hand_joint_features=cfg.agent.grasp_evaluator_hand_joint_features,
            mask_constant_features=cfg.agent.grasp_evaluator_mask_constant_features,
        ).cuda()
        width = len(predictor.center) - len(original["center"])
        original["center"] = F.pad(original["center"], (0, width))
        original["deviation"] = F.pad(original["deviation"], (0, width), value=1)
        original["network.0.weight"] = F.pad(original["network.0.weight"], (0, width))
        predictor.load_state_dict(original, strict=True)
    inputs = torch.cat([features, torch.cat([predictor.encode_action(batch) for batch in actions.split(32)])], -1)
    meshes = [splits["train"][sid].meshes[splits["train"][sid].target_actor_id].collision_mesh_path for sid in ids]
    validation_ids = set(saved["scorer_validation_ids"])
    assert len(validation_ids) == len(saved["scorer_validation_ids"]) and validation_ids <= set(ids)
    train_ids = torch.tensor([i for i, sid in enumerate(ids) if sid not in validation_ids], device="cuda:0")
    valid_ids = torch.tensor([i for i, sid in enumerate(ids) if sid in validation_ids], device="cuda:0")
    assert len(train_ids) > 0 and len(valid_ids) >= validation_scenes
    assert not {meshes[i] for i in train_ids.tolist()} & {meshes[i] for i in valid_ids.tolist()}
    if observations is not None:
        assert ranking_weight > 0 and cfg.agent.generative_model == "flow_transformer"
        assert cfg.agent.vision_encoder_global_feature and actions.shape[1] == 64
        dataset = GraspDataset(
            replace(cfg.dataset, return_actions=False, return_graspness=False),
            str(observations),
            cfg.scene_description_path,
            split="train",
        )
        lookup = {sid: index for index, sid in enumerate(dataset.scene_ids)}
        assert set(ids) <= set(lookup)
        observed = []
        local_tokens = []
        local_conditions = []
        anchor_states = []
        if local_features:
            policy.cfg = replace(cfg.agent, grasp_evaluator_guidance_steps=0, grasp_evaluator_local_features=True)
        for start in range(0, len(ids), 16 if local_features else 1):
            batch = []
            for sid in ids[start : start + (16 if local_features else 1)]:
                observation = dataset[lookup[sid]]["observations"]
                keys = (
                    ("rgb", "qpos", "camera_poses") if cfg.agent.vision_encoder.startswith("rgb") else ("pointcloud",)
                )
                if cfg.agent.vision_encoder == "pointnet_state":
                    keys = ("pointcloud", "qpos", "camera_poses")
                if anchor_state and "qpos" not in keys:
                    keys = (*keys, "qpos")
                observation = {key: observation[key] for key in keys}
                if cfg.agent.vision_encoder.startswith("rgb"):
                    observation["rgb"] = F.interpolate(
                        observation["rgb"][None].float(), size=(128, 128), mode="bilinear", align_corners=False
                    )[0]
                batch.append(observation)
            if local_features:
                obs = {key: value.cuda() for key, value in torch.utils.data.default_collate(batch).items()}
                captured = []
                hook = policy.generator.cond_proj.register_forward_pre_hook(
                    lambda module, arguments, captured=captured: captured.append(arguments[0])
                )
                with torch.no_grad():
                    policy.get_action(obs)
                hook.remove()
                assert len(captured) == 8 and all(torch.equal(value, captured[0]) for value in captured)
                local_tokens.append(obs["critic_tokens"].cpu())
                local_conditions.append(captured[0])
                if anchor_state:
                    anchor_states.append(
                        predictor.encode_action(
                            obs["qpos"][:, -1, None].expand(-1, cfg.agent.num_queries, -1).flatten(1)
                        ).unflatten(-1, (cfg.agent.num_queries, -1))[:, :1]
                    )
            else:
                observed.extend(batch)
        dataset.h5_file.close()
        if local_features:
            local_tokens = torch.cat(local_tokens)
            assert local_tokens.shape == (len(ids), 64, features.shape[-1])
            features = torch.cat(local_conditions)[:, None].expand(-1, actions.shape[1], -1)
            inputs = torch.cat([features, inputs[..., features.shape[-1] :]], -1)
            if anchor_state:
                anchor_states = torch.cat(anchor_states)
                assert anchor_states.shape == (len(ids), 1, len(predictor.center) - inputs.shape[-1])
                assert not {ids[i] for i in train_ids.tolist()} & validation_ids
                predictor.center[inputs.shape[-1] :].copy_(anchor_states[train_ids, 0].mean(0))
                predictor.deviation[inputs.shape[-1] :].copy_(anchor_states[train_ids, 0].std(0).clamp_min(1e-4))
            seed_everything(seed)
    train_x, valid_x = inputs[train_ids], inputs[valid_ids]
    train_y, valid_y = rewards[train_ids], rewards[valid_ids]
    assert train_y.any() and (train_y == 0).any()
    flat_train_x, flat_train_y = train_x.flatten(0, 1), train_y.flatten()
    if not cfg.agent.grasp_evaluator_guidance_steps:
        predictor.center.copy_(flat_train_x.mean(0))
        predictor.deviation.copy_(flat_train_x.std(0).clamp_min(1e-4))
    policy.grasp_evaluator = predictor
    policy.cfg = replace(
        cfg.agent,
        grasp_evaluator_guidance_steps=guidance_steps,
        grasp_evaluator_guidance_radius=guidance_radius,
        grasp_evaluator_local_features=local_features,
        grasp_evaluator_anchor_state=anchor_state,
    )
    learning_rate = 1e-4 if cfg.agent.grasp_evaluator_guidance_steps else 1e-3
    optimizer = torch.optim.AdamW(
        policy.parameters() if observations is not None and not local_features else predictor.parameters(),
        lr=learning_rate,
        weight_decay=1e-3,
    )
    provenance = {
        "experiment": experiment,
        "policy_config": asdict(policy.cfg),
        "processed_data_path": source["processed_data_path"],
        "training": {
            **source["training"],
            "seed": seed,
            "iterations": iterations,
            "batch_size": 1024,
            "lr": learning_rate,
            "weight_decay": 1e-3,
            "checkpoint_selection": "selected_success_64_object_disjoint" if ranking_weight else "bce_object_disjoint",
            "ranking_weight": ranking_weight,
            "source_train_ids": source["training"]["train_ids"],
            "train_ids": sorted(set(source["training"]["train_ids"]) | {ids[i] for i in train_ids.tolist()}),
            "scorer_train_ids": [ids[i] for i in train_ids.tolist()],
            "scorer_validation_ids": [ids[i] for i in valid_ids.tolist()],
            "reserved_validation_ids": saved["reserved_validation_ids"],
            **(
                {
                    "joint_updates": iterations,
                    "joint_train_ids": [ids[i] for i in train_ids.tolist()],
                    "joint_observations": str(observations),
                    "joint_objective": "policy_loss_positive_only + BCE_all64 + ranking",
                }
                if observations is not None and not local_features
                else {}
            ),
            **(
                {
                    "local_feature_updates": iterations,
                    "local_feature_observations": str(observations),
                    "local_feature_tokens": 64,
                    "local_feature_actor_frozen": True,
                    "local_feature_global_reencoded": True,
                }
                if local_features
                else {}
            ),
            **(
                {
                    "anchor_state_updates": iterations,
                    "anchor_state_norm_train_ids": [ids[i] for i in train_ids.tolist()],
                    "anchor_state_current_qpos": True,
                }
                if anchor_state
                else {}
            ),
        },
    }
    best = (float("inf"), float("inf")) if ranking_weight else float("inf")
    for iteration in range(iterations):
        if observations is not None and not local_features:
            policy.train()
        predictor.train()
        count = len(train_y) if ranking_weight else len(flat_train_y)
        batch_size = 1024 // actions.shape[1] if ranking_weight else 1024
        assert batch_size > 0 and (not ranking_weight or 1024 % actions.shape[1] == 0)
        indices = torch.randint(count, (batch_size,), device="cuda:0")
        policy_loss = rewards.new_zeros(())
        if ranking_weight:
            labels = train_y[indices].reshape(-1, 64)
            batch = train_x[indices]
            if observations is not None and not local_features:
                scene_indices = train_ids[indices]
                obs = {
                    key: value.cuda()
                    for key, value in torch.utils.data.default_collate(
                        [observed[index] for index in scene_indices.tolist()]
                    ).items()
                }
                captured = []
                hook = policy.generator.cond_proj.register_forward_pre_hook(
                    lambda module, arguments, captured=captured: captured.append(arguments[0])
                )
                policy(obs, actions[scene_indices, :1].unflatten(-1, (cfg.agent.num_queries, -1)))
                hook.remove()
                assert len(captured) == 1 and captured[0].shape == (batch_size, features.shape[-1])
                batch = torch.cat([captured[0][:, None].expand(-1, 64, -1), batch[..., features.shape[-1] :]], -1)
                positive = labels.any(1)
                if positive.any():
                    selected = torch.multinomial(labels[positive], 1).squeeze(1)
                    successful = actions[scene_indices[positive], selected][:, None]
                    policy_loss = policy(
                        {key: value[positive] for key, value in obs.items()},
                        successful.unflatten(-1, (cfg.agent.num_queries, -1)),
                    )["loss"]
            logits = predictor(
                batch[..., : features.shape[-1]],
                batch[..., features.shape[-1] :],
                local_tokens[train_ids[indices].cpu()].cuda() if local_features else None,
                encoded_action=True,
                anchor_state=anchor_states[train_ids[indices]] if anchor_state else None,
            ).reshape_as(labels)
            valid = labels.any(1) & (labels == 0).any(1)
            positive = logits[valid].masked_fill(labels[valid] == 0, -torch.inf).max(1).values
            negative = logits[valid].masked_fill(labels[valid] > 0, -torch.inf).max(1).values
            ranking = F.softplus(negative - positive).sum() / valid.sum().clamp_min(1)
        else:
            labels = flat_train_y[indices]
            logits = predictor.network(predictor.normalize(flat_train_x[indices])).squeeze(-1)
            ranking = 0.0
        loss = F.binary_cross_entropy_with_logits(logits, labels) + ranking_weight * ranking + policy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (iteration + 1) % 100 and iteration + 1 != iterations:
            continue
        predictor.eval()
        with torch.no_grad():
            batch = valid_x
            if observations is not None and not local_features:
                policy.eval()
                captured = []
                hook = policy.generator.cond_proj.register_forward_pre_hook(
                    lambda module, arguments, captured=captured: captured.append(arguments[0])
                )
                for scene_indices in valid_ids.split(32):
                    obs = {
                        key: value.cuda()
                        for key, value in torch.utils.data.default_collate(
                            [observed[index] for index in scene_indices.tolist()]
                        ).items()
                    }
                    policy(obs, actions[scene_indices, :1].unflatten(-1, (cfg.agent.num_queries, -1)))
                hook.remove()
                encoded = torch.cat(captured)
                assert encoded.shape == (len(valid_ids), features.shape[-1])
                batch = torch.cat([encoded[:, None].expand(-1, 64, -1), batch[..., features.shape[-1] :]], -1)
            logits = torch.cat(
                [
                    predictor(
                        chunk[..., : features.shape[-1]],
                        chunk[..., features.shape[-1] :],
                        local_tokens[valid_ids[start : start + len(chunk)].cpu()].cuda() if local_features else None,
                        encoded_action=True,
                        anchor_state=anchor_states[valid_ids[start : start + len(chunk)]] if anchor_state else None,
                    )
                    for start, chunk in zip(range(0, len(batch), 32), batch.split(32), strict=True)
                ]
            ).reshape_as(valid_y)
            validation_loss = F.binary_cross_entropy_with_logits(logits, valid_y).item()
            validation_selection = validation_loss
            if ranking_weight:
                labels, scores = valid_y.reshape(-1, 64), logits.reshape(-1, 64)
                selected_success = labels.gather(1, scores.argmax(1, keepdim=True)).mean().item()
                valid = labels.any(1) & (labels == 0).any(1)
                positive = scores[valid].masked_fill(labels[valid] == 0, -torch.inf).max(1).values
                negative = scores[valid].masked_fill(labels[valid] > 0, -torch.inf).max(1).values
                validation_ranking = (F.softplus(negative - positive).sum() / valid.sum().clamp_min(1)).item()
                validation_selection = (-selected_success, validation_loss + ranking_weight * validation_ranking)
        if validation_selection < best:
            best = validation_selection
            torch.save(
                {
                    "agent": policy.state_dict(),
                    "iteration": iteration + 1,
                    "validation_bce": validation_loss,
                    **(
                        {"validation_ranking": validation_ranking, "validation_selected_success": selected_success}
                        if ranking_weight
                        else {}
                    ),
                    **provenance,
                },
                output,
            )
        row = {"iteration": iteration + 1, "validation_bce": validation_loss}
        if observations is not None and not local_features:
            row["policy_loss_positive_only"] = policy_loss.item()
        if ranking_weight:
            row["validation_ranking"] = validation_ranking
            row["validation_selected_success"] = selected_success
        print(json.dumps(row), flush=True)
    metric = "best_validation_selected_success" if ranking_weight else "best_validation_bce"
    print(json.dumps({"output": str(output), metric: -best[0] if ranking_weight else best}), flush=True)


if __name__ == "__main__":
    tyro.cli(main)
