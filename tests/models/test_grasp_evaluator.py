"""Tests for `graspkit.models.grasp_evaluator` and evaluator-guided policy sampling."""

from dataclasses import replace

import pytest
import torch

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot import build_robot_config
from graspkit.models.cvae import CVAE
from graspkit.models.grasp_evaluator import GraspEvaluator
from graspkit.models.grasp_policy import GraspPolicy
from graspkit.robot.grasp_robot import GraspRobot
from scripts.stages.train.mirror_checkpoint import mirror_floating_pointnet_model


@pytest.mark.hf_assets
class TestGraspEvaluator:
    def test_hand_scorer_preserves_geometry_and_explicit_constant_mask(self):
        robot = build_robot_config("arx_x5_arm")
        helper = GraspRobot.from_config(robot)
        all_joints = GraspEvaluator(4, robot, 3)
        hand = GraspEvaluator(4, robot, 3, hand_joint_features=True, mask_constant_features=False)
        actions = helper.init_qpos.repeat(2, 3).reshape(2, -1)
        full = all_joints.encode_action(actions).reshape(2, 3, -1)
        encoded = hand.encode_action(actions).reshape(2, 3, -1)
        geometry = full.shape[-1] - helper.num_actuated_joints
        torch.testing.assert_close(encoded[..., :geometry], full[..., :geometry], atol=0, rtol=0)
        torch.testing.assert_close(
            encoded[..., geometry:], full[..., geometry:][..., helper.hand_joint_indices], atol=0, rtol=0
        )
        assert hand.network[0].in_features == 4 + 3 * (geometry + len(helper.hand_joint_indices))
        for predictor in (all_joints, hand):
            predictor.deviation[0] = 1e-4
            x = predictor.center.clone()
            x[0] += 1
            actual = predictor.normalize(x)
            assert actual[0] == (0 if predictor.mask_constant_features else 10000)

    def test_constant_replay_features_cannot_create_arbitrary_scores(self):
        torch.manual_seed(0)
        predictor = GraspEvaluator(4, build_robot_config("arx_x5_arm"), 3).eval().requires_grad_(False)
        predictor.center[:2] = torch.tensor([2.0, -3.0])
        predictor.deviation[:2] = 1e-4
        condition = torch.zeros(2, 4)
        condition[:, :2] = predictor.center[:2]
        condition[1, :2] += torch.tensor([1000.0, -1000.0])
        condition.requires_grad_(True)
        action = torch.zeros(2, predictor.center.numel() - 4)
        scores = predictor(condition, action, encoded_action=True)
        torch.testing.assert_close(scores[0], scores[1], atol=0, rtol=0)
        gradient = torch.autograd.grad(scores.sum(), condition)[0]
        assert (gradient[:, :2] == 0).all()
        assert gradient[:, 2:].abs().sum() > 0
        changed = condition.detach().clone()
        changed[1, 2:] = 1000
        assert not torch.equal(scores.detach(), predictor(changed, action, encoded_action=True))

    @pytest.mark.parametrize("inference_mode", [False, True])
    @pytest.mark.parametrize("robot_name", ["arx_x5_arm", "allegro_right"])
    def test_refinement_is_bounded_and_rng_free(self, inference_mode: bool, robot_name: str):
        torch.manual_seed(0)
        robot = build_robot_config(robot_name)
        action_dim = GraspRobot.from_config(robot).num_actuated_joints + (6 if robot.base_type == "floating" else 0)
        cvae = CVAE(x_dim=action_dim * 3, cond_dim=4, latent_dim=3, hidden_dims=[8])
        predictor = GraspEvaluator(4, robot, 3).eval()
        helper = GraspRobot.from_config(robot)
        limits = helper.actuated_joint_limits_torch
        joints = helper.init_qpos.clamp(limits[:, 0], limits[:, 1])
        hand = helper.hand_joint_indices[0]
        joints[hand] = limits[hand, 1]
        executable = torch.zeros(1, 3, action_dim)
        executable[..., -helper.num_actuated_joints :] = joints
        outside = executable.clone()
        outside[..., action_dim - helper.num_actuated_joints + hand] += 0.25
        outside.requires_grad_(True)
        encoded = predictor.encode_action(outside.flatten(1))
        phase_width = encoded.shape[-1] // 3
        assert phase_width == 3 * len(robot.fingertip_link_names) + 6 + helper.num_actuated_joints
        torch.testing.assert_close(
            encoded.unflatten(-1, (3, phase_width))[..., -helper.num_actuated_joints :],
            executable[..., -helper.num_actuated_joints :],
            atol=0,
            rtol=0,
        )
        assert predictor.network[0].in_features == 4 + encoded.shape[-1]
        torch.testing.assert_close(encoded, predictor.encode_action(executable.flatten(1)), atol=0, rtol=0)
        gradient = torch.autograd.grad(encoded.sum(), outside)[0]
        assert (gradient[..., action_dim - helper.num_actuated_joints + hand] == 0).all()
        condition, latent = torch.randn(5, 4), torch.randn(5, 3)
        decoded_latents = []
        handle = cvae.decoder.register_forward_pre_hook(
            lambda module, inputs: decoded_latents.append(inputs[0][:, :3].detach())
        )
        state = torch.get_rng_state()
        with torch.inference_mode(inference_mode):
            result = predictor.refine(cvae, latent, condition, steps=10, radius=0.5)
        handle.remove()
        assert torch.equal(state, torch.get_rng_state())
        assert len(decoded_latents) == 11
        assert all(((value - latent).norm(dim=-1) <= 0.5 + 1e-6).all() for value in decoded_latents)
        assert torch.isfinite(result).all() and not result.requires_grad
        assert all(parameter.grad is None for parameter in [*cvae.parameters(), *predictor.parameters()])

    @pytest.mark.parametrize("use_mean", [False, True])
    @pytest.mark.parametrize("robot_name", ["arx_x5", "panda_allegro", "agilex_aloha", "yam", "xarm7_sharpa"])
    @pytest.mark.parametrize(
        "generative_model,learned_prior",
        [("cvae", False), ("cvae", True), ("flow", False), ("flow_transformer", False)],
    )
    @pytest.mark.parametrize("global_feature", [False, True])
    def test_guided_policy_sampling_and_checkpoint(
        self, use_mean: bool, robot_name: str, generative_model: str, learned_prior: bool, global_feature: bool
    ):
        pytest.importorskip("torchvision")  # the rgb_resnet_views encoder below
        torch.manual_seed(0)
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_feature_dim=32,
            vision_encoder_global_feature=global_feature,
            use_mean=use_mean,
            cvae_learned_prior=learned_prior,
            generative_model=generative_model,
            cvae_hidden_dim=8,
            cvae_temperature=0.7,
        )
        robot = build_experiment_config(robot_name + "_tabletop").robot_config
        action_dim = GraspRobot.from_config(robot).num_actuated_joints
        baseline = GraspPolicy(cfg, action_dim, robot).eval()
        if not global_feature:
            # Negative grasp logits must still rank object points above the background.
            for parameter in [*baseline.objectness_decoder.parameters(), *baseline.graspness_decoder.parameters()]:
                parameter.data.zero_()
            baseline.objectness_decoder[0].weight.data[0, 0] = 1
            baseline.objectness_decoder[2].weight.data[0, 0] = 1
            baseline.objectness_decoder[2].bias.data.fill_(-0.5)
            baseline.graspness_decoder[2].bias.data.fill_(-1)
        guided_cfg = replace(cfg, grasp_evaluator_guidance_steps=5)
        guided = GraspPolicy(guided_cfg, action_dim, robot).eval()
        guided.vision_encoder.load_state_dict(baseline.vision_encoder.state_dict(), strict=True)
        guided.generator.load_state_dict(baseline.generator.state_dict(), strict=True)
        if not global_feature:
            guided.objectness_decoder.load_state_dict(baseline.objectness_decoder.state_dict(), strict=True)
            guided.graspness_decoder.load_state_dict(baseline.graspness_decoder.state_dict(), strict=True)
        restored = GraspPolicy(guided_cfg, action_dim, robot).eval()
        restored.load_state_dict(guided.state_dict(), strict=True)
        point_hooks = []
        if not global_feature:
            local_features = torch.zeros(3, 32, 32)
            local_features[:, 16:, 0] = 1
            for policy in [baseline, guided, restored]:
                point_hooks.append(
                    policy.vision_encoder.register_forward_hook(lambda module, inputs, value: local_features)
                )
        with pytest.raises(AssertionError, match="before adding a grasp evaluator"):
            mirror_floating_pointnet_model(guided)
        obs = {"pointcloud": torch.randn(3, 32, 3)}
        if global_feature:
            expected_conditions = baseline.vision_encoder(obs["pointcloud"] * cfg.pointcloud_scale).repeat_interleave(
                2, dim=0
            )
        baseline_latents, guided_latents, conditions = [], [], []
        condition_hook = (
            baseline.generator.cond_proj if generative_model == "flow_transformer" else baseline.generator.decoder
        ).register_forward_pre_hook(
            lambda module, inputs: conditions.append(
                inputs[0].detach()
                if generative_model == "flow_transformer"
                else inputs[0][:, baseline.generator.latent_dim : baseline.generator.latent_dim + 32].detach()
            )
        )
        baseline_hook = baseline.generator.decoder.register_forward_pre_hook(
            lambda module, inputs: baseline_latents.append(
                inputs[0][:, : baseline.generator.latent_dim].detach().clone()
            )
        )
        guided_hook = guided.generator.decoder.register_forward_pre_hook(
            lambda module, inputs: guided_latents.append(inputs[0][:, : guided.generator.latent_dim].detach().clone())
        )
        torch.manual_seed(7)
        baseline.get_action(obs, num_samples_per_scene=2)
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(7)
        with torch.inference_mode():
            actual = guided.get_action(obs, num_samples_per_scene=2)
        baseline_hook.remove()
        guided_hook.remove()
        condition_hook.remove()
        if not global_feature:
            assert torch.equal(conditions[0][:, 0], torch.ones(6))
        else:
            torch.testing.assert_close(conditions[0], expected_conditions, atol=0, rtol=0)
        torch.testing.assert_close(baseline_latents[0], guided_latents[0], atol=0, rtol=0)
        assert torch.equal(baseline_rng, torch.get_rng_state())
        torch.manual_seed(7)
        torch.testing.assert_close(actual, restored.get_action(obs, num_samples_per_scene=2), atol=0, rtol=0)
        for hook in point_hooks:
            hook.remove()
        assert actual.shape == (3, 2, 3, action_dim) and torch.isfinite(actual).all()
        if global_feature and generative_model == "flow_transformer" and not use_mean:
            guided.train()
            captured = []
            hook = guided.generator.cond_proj.register_forward_pre_hook(
                lambda module, inputs: captured.append(inputs[0])
            )
            guided(obs, actual[:, :1].detach())
            hook.remove()
            assert len(captured) == 1 and captured[0].requires_grad
            captured[0].retain_grad()
            logits = guided.grasp_evaluator(
                captured[0][:, None].expand(-1, 2, -1), actual.detach().flatten(-2)
            ).squeeze(-1)
            labels = logits.new_tensor([[1, 0], [0, 0], [0, 0]])
            classification = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            gradients = torch.autograd.grad(classification, guided.vision_encoder.parameters(), retain_graph=True)
            assert sum(value.abs().sum() for value in gradients) > 0
            # Only the first observation has successful actions; zero pools still train the scorer.
            reconstruction = guided({key: value[:1] for key, value in obs.items()}, actual[:1, :1].detach())["loss"]
            center, deviation = guided.generator.action_center.clone(), guided.generator.action_scale.clone()
            optimizer = torch.optim.AdamW(guided.parameters(), lr=1e-4)
            optimizer.zero_grad()
            (classification + reconstruction).backward()
            assert captured[0].grad[1:].abs().sum() > 0
            assert sum(parameter.grad.abs().sum() for parameter in guided.generator.parameters()) > 0
            optimizer.step()
            torch.testing.assert_close(guided.generator.action_center, center, atol=0, rtol=0)
            torch.testing.assert_close(guided.generator.action_scale, deviation, atol=0, rtol=0)
            restored.load_state_dict(guided.state_dict(), strict=True)
            restored.eval()
            guided.eval()
            torch.manual_seed(9)
            expected = guided.get_action(obs, num_samples_per_scene=2)
            torch.manual_seed(9)
            torch.testing.assert_close(expected, restored.get_action(obs, num_samples_per_scene=2), atol=0, rtol=0)
            for modality in ("pointnet", "rgb_resnet_views"):
                legacy_cfg = replace(guided_cfg, vision_encoder=modality, vision_encoder_feature_dim=512)
                legacy = GraspPolicy(legacy_cfg, action_dim, robot).eval().requires_grad_(False)
                token_cfg = replace(legacy_cfg, grasp_evaluator_local_features=True)
                token_policy = GraspPolicy(token_cfg, action_dim, robot).eval().requires_grad_(False)
                token_policy.vision_encoder.load_state_dict(legacy.vision_encoder.state_dict(), strict=True)
                token_policy.generator.load_state_dict(legacy.generator.state_dict(), strict=True)
                if modality == "rgb_resnet_views":
                    token_policy.rgb_conditioning.load_state_dict(legacy.rgb_conditioning.state_dict(), strict=True)
                token_policy.grasp_evaluator.network.load_state_dict(
                    legacy.grasp_evaluator.network.state_dict(), strict=True
                )
                token_obs = (
                    {"pointcloud": torch.randn(2, 64, 3)}
                    if modality == "pointnet"
                    else {
                        "rgb": torch.randint(256, (2, 12, 128, 128)),
                        "qpos": torch.zeros(2, 2, action_dim),
                        "camera_poses": torch.zeros(2, 4, 1, 7),
                    }
                )
                frozen = {
                    key: value.clone()
                    for key, value in token_policy.state_dict().items()
                    if not key.startswith("grasp_evaluator.")
                }
                conditions, maps = [], []
                hook = token_policy.generator.cond_proj.register_forward_pre_hook(
                    lambda module, arguments: conditions.append(arguments[0])
                )
                map_hook = (
                    token_policy.vision_encoder.conv3 if modality == "pointnet" else token_policy.vision_encoder.avgpool
                ).register_forward_hook(lambda module, arguments, output: maps.append(output))
                torch.manual_seed(11)
                original = legacy.get_action(token_obs, 2)
                original_rng = torch.get_rng_state()
                torch.manual_seed(11)
                with torch.inference_mode():
                    adapted = token_policy.get_action(token_obs, 2)
                hook.remove()
                map_hook.remove()
                torch.testing.assert_close(adapted, original, atol=1e-6, rtol=1e-6)
                assert torch.equal(torch.get_rng_state(), original_rng)
                tokens = token_obs["critic_tokens"].clone()
                assert tokens.shape == (2, 64, 512) and not tokens.requires_grad
                if modality == "pointnet":
                    torch.testing.assert_close(
                        tokens, token_policy.vision_encoder.feat_linear(maps[0].transpose(1, 2)), atol=1e-6, rtol=1e-6
                    )
                else:
                    raw = maps[0].reshape(2, 4, 512, 4, 4).permute(0, 1, 3, 4, 2).reshape(2, 64, 512)
                    positional = tokens - raw
                    torch.testing.assert_close(positional[0], positional[1])
                    assert len(torch.unique(positional[0].round(decimals=5), dim=0)) == 64
                condition = conditions[0].clone().unflatten(0, (2, 2))
                action = adapted.clone().flatten(-2)
                critic = token_policy.grasp_evaluator.requires_grad_(True)
                critic.train()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    critic(condition, action, tokens).squeeze(-1), torch.tensor([[1.0, 0.0], [0.0, 0.0]])
                )
                optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
                optimizer.zero_grad()
                loss.backward()
                assert critic.attention.out_proj.weight.grad.abs().sum() > 0
                optimizer.step()
                assert all(torch.equal(value, token_policy.state_dict()[key]) for key, value in frozen.items())
                critic.eval()
                torch.testing.assert_close(
                    critic(condition, action, tokens),
                    critic(condition, critic.encode_action(action), tokens, encoded_action=True),
                    atol=0,
                    rtol=0,
                )
                with pytest.raises(AssertionError):
                    critic(condition, action)
                token_policy.requires_grad_(False)
                token_restored = GraspPolicy(token_cfg, action_dim, robot).eval().requires_grad_(False)
                token_restored.load_state_dict(token_policy.state_dict(), strict=True)
                torch.manual_seed(13)
                with torch.inference_mode():
                    expected = token_policy.get_action(token_obs, 2)
                torch.manual_seed(13)
                with torch.inference_mode():
                    reloaded = token_restored.get_action(token_obs, 2)
                torch.testing.assert_close(reloaded, expected, atol=0, rtol=0)
                anchor_cfg = replace(token_cfg, grasp_evaluator_anchor_state=True)
                anchored = GraspPolicy(anchor_cfg, action_dim, robot).eval().requires_grad_(False)
                state = token_policy.state_dict()
                prefix = len(token_policy.grasp_evaluator.center)
                width = len(anchored.grasp_evaluator.center) - prefix
                state["grasp_evaluator.center"] = torch.nn.functional.pad(state["grasp_evaluator.center"], (0, width))
                state["grasp_evaluator.deviation"] = torch.nn.functional.pad(
                    state["grasp_evaluator.deviation"], (0, width), value=1
                )
                state["grasp_evaluator.network.0.weight"] = torch.nn.functional.pad(
                    state["grasp_evaluator.network.0.weight"], (0, width)
                )
                anchored.load_state_dict(state, strict=True)
                helper = GraspRobot.from_config(robot)
                qpos = helper.init_qpos[None, None].expand(2, 2, -1).clone()
                hand = helper.hand_joint_indices[0]
                limits = helper.actuated_joint_limits_torch
                qpos[:, -1, hand] = torch.lerp(limits[hand, 0], limits[hand, 1], torch.tensor([0.25, 0.75]))
                token_obs["qpos"] = qpos
                anchor = anchored.grasp_evaluator.encode_action(
                    qpos[:, -1:].expand(-1, anchor_cfg.num_queries, -1).flatten(1)
                ).unflatten(-1, (anchor_cfg.num_queries, -1))[:, :1]
                anchored.grasp_evaluator.center[prefix:].copy_(anchor[:, 0].mean(0))
                anchored.grasp_evaluator.deviation[prefix:].copy_(anchor[:, 0].std(0).clamp_min(1e-4))
                torch.testing.assert_close(anchored.grasp_evaluator.center[:prefix], critic.center, atol=0, rtol=0)
                torch.testing.assert_close(
                    anchored.grasp_evaluator.deviation[:prefix], critic.deviation, atol=0, rtol=0
                )
                assert (anchored.grasp_evaluator.network[0].weight[:, prefix:] == 0).all()
                seen = []
                handle = anchored.grasp_evaluator.register_forward_pre_hook(
                    lambda module, arguments, keywords: seen.append(keywords["anchor_state"].clone()), with_kwargs=True
                )
                torch.manual_seed(17)
                with torch.inference_mode():
                    original = token_policy.get_action(token_obs, 2)
                original_rng = torch.get_rng_state()
                torch.manual_seed(17)
                with torch.inference_mode():
                    adapted = anchored.get_action(token_obs, 2)
                handle.remove()
                torch.testing.assert_close(adapted, original, atol=1e-6, rtol=1e-6)
                assert torch.equal(original_rng, torch.get_rng_state())
                assert len(seen) == anchor_cfg.grasp_evaluator_guidance_steps
                assert all(torch.equal(value, anchor) for value in seen)
                frozen = {
                    key: value.clone()
                    for key, value in anchored.state_dict().items()
                    if not key.startswith("grasp_evaluator.")
                }
                anchor_critic = anchored.grasp_evaluator.requires_grad_(True).train()
                scores = anchor_critic(condition, action, tokens, anchor_state=anchor).squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    scores, torch.tensor([[1.0, 0.0], [0.0, 0.0]])
                )
                optimizer = torch.optim.AdamW(anchor_critic.parameters(), lr=1e-3)
                optimizer.zero_grad()
                loss.backward()
                assert anchor_critic.network[0].weight.grad[:, prefix:].abs().sum() > 0
                assert all(
                    parameter.grad is None
                    for parameter in [*anchored.vision_encoder.parameters(), *anchored.generator.parameters()]
                )
                optimizer.step()
                assert all(torch.equal(value, anchored.state_dict()[key]) for key, value in frozen.items())
                anchor_critic.eval()
                torch.testing.assert_close(
                    anchor_critic(condition, action, tokens, anchor_state=anchor),
                    anchor_critic(
                        condition, anchor_critic.encode_action(action), tokens, encoded_action=True, anchor_state=anchor
                    ),
                    atol=0,
                    rtol=0,
                )
                assert not torch.equal(
                    anchor_critic(condition, action, tokens, anchor_state=anchor),
                    anchor_critic(condition, action, tokens, anchor_state=anchor.flip(0)),
                )
                with pytest.raises(AssertionError):
                    anchor_critic(condition, action, tokens)
                anchored.requires_grad_(False)
                anchor_restored = GraspPolicy(anchor_cfg, action_dim, robot).eval().requires_grad_(False)
                anchor_restored.load_state_dict(anchored.state_dict(), strict=True)
                torch.manual_seed(19)
                with torch.inference_mode():
                    expected = anchored.get_action(token_obs, 2)
                torch.manual_seed(19)
                with torch.inference_mode():
                    reloaded = anchor_restored.get_action(token_obs, 2)
                torch.testing.assert_close(reloaded, expected, atol=0, rtol=0)
