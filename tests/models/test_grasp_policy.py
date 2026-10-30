"""Tests for `graspkit.models.grasp_policy`."""

from types import SimpleNamespace

import pytest
import torch
from robokit.xform.utils import rotation_6d_to_matrix
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_nearest_euler_angles
from torch.nn import functional as F

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot import build_robot_config
from graspkit.configs.robot_config import RobotConfig
from graspkit.models.grasp_policy import GraspPolicy
from graspkit.robot.grasp_robot import GraspRobot


class TestGraspPolicy:
    @pytest.mark.parametrize("encoder", ["pointnet", "pointnet_state", "rgb", "rgb_resnet", "rgb_resnet_views"])
    @pytest.mark.parametrize("model", ["cvae", "flow", "flow_transformer"])
    def test_policy_modalities_and_generators(self, encoder: str, model: str):
        if encoder.startswith("rgb_resnet"):
            pytest.importorskip("torchvision")
        torch.manual_seed(5)
        cfg = GraspPolicyConfig(
            vision_encoder=encoder,
            generative_model=model,
            vision_encoder_feature_dim=32,
            cvae_latent_dim=8,
            cvae_hidden_dim=32,
            use_mean=False,
        )
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="fixed")
        policy = GraspPolicy(cfg, 8, robot)
        obs = (
            {"rgb": torch.randint(0, 256, (2, 12 if encoder == "rgb_resnet_views" else 9, 32, 48), dtype=torch.uint8)}
            if encoder.startswith("rgb")
            else {"pointcloud": torch.randn(2, 32, 3)}
        )
        if encoder in ("pointnet_state", "rgb_resnet_views"):
            obs["qpos"] = torch.randn(2, 2, 8)
            obs["camera_poses"] = torch.randn(2, 4, 1, 7)
        target = torch.randn(2, 3, 3, 8)
        reconstructed = []
        if model != "cvae":
            scale = torch.ones(3, 8)
            scale[:, 0] = torch.tensor([0.001, 0.012, 0.001])
            with torch.no_grad():
                policy.generator.action_scale.copy_(scale.flatten())
            hook = policy.generator.register_forward_hook(
                lambda module, inputs, output: reconstructed.append(output[0])
            )
        losses = policy(obs, target)
        if model != "cvae":
            hook.remove()
            gradient = torch.autograd.grad(losses["qpos"], reconstructed[0], retain_graph=True)[0]
            residual = reconstructed[0].reshape_as(target) - target
            torch.testing.assert_close(
                gradient.reshape_as(target)[..., 0], 2 * residual[..., 0] / (target.numel() * 0.012**2)
            )
            torch.testing.assert_close(policy.generator.action_scale, scale.flatten())
        loss = losses["loss"]
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.generator.parameters())
        policy.eval()
        torch.manual_seed(6)
        with torch.inference_mode():
            actions = policy.get_action(obs, 4)
        assert actions.shape == (2, 4, 3, 8) and torch.isfinite(actions).all()
        if model != "cvae":
            torch.testing.assert_close(policy.generator.action_scale, scale.flatten())
        if encoder == "pointnet_state":
            assert policy.rgb_conditioning.weight.grad[:, 32:].abs().sum() > 0
            with torch.no_grad():
                policy.rgb_conditioning.weight[:, 32:].fill_(0.02)
            torch.manual_seed(6)
            reference = policy.get_action(obs, 4)
            torch.manual_seed(6)
            changed = policy.get_action({**obs, "qpos": obs["qpos"] + 1}, 4)
            assert not torch.allclose(reference, changed)
            restored = GraspPolicy(cfg, 8, robot).eval()
            restored.load_state_dict(policy.state_dict(), strict=True)
            torch.manual_seed(6)
            torch.testing.assert_close(reference, restored.get_action(obs, 4))
        if encoder.startswith("rgb"):
            torch.manual_seed(6)
            with torch.inference_mode():
                repeated = policy.get_action({**obs, "pointcloud": torch.full((2, 32, 3), float("nan"))}, 4)
            torch.testing.assert_close(actions, repeated)
        if encoder == "rgb_resnet_views":
            torch.manual_seed(6)
            with torch.inference_mode():
                reversed_views = policy.get_action(
                    {**obs, "rgb": obs["rgb"].unflatten(1, (4, 3)).flip(1).flatten(1, 2)}, 4
                )
            torch.testing.assert_close(actions, reversed_views)


class TestPolicyLosses:
    @pytest.mark.parametrize("model", ["cvae", "flow_transformer"])
    def test_phase_std_mse_uses_each_phase_and_preserves_other_losses(self, model: str):
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_feature_dim=32,
            cvae_latent_dim=8,
            cvae_hidden_dim=32,
            generative_model=model,
            normalize_cvae_actions=model == "cvae",
            reconstruction_loss="phase_std_mse",
        )
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf")
        policy = GraspPolicy(cfg, 8, robot)
        scale = torch.ones(3, 8)
        scale[:, 0] = torch.tensor([0.001, 0.012, 0.1])
        policy.generator.action_scale.copy_(scale.flatten())
        actions = torch.randn(2, 3, 3, 8)
        captured = []
        hook = policy.generator.register_forward_hook(lambda module, arguments, output: captured.append(output[0]))
        losses = policy({"pointcloud": torch.randn(2, 32, 3)}, actions)
        hook.remove()
        prediction = captured[0].reshape_as(actions)
        torch.testing.assert_close(losses["qpos"], F.mse_loss(prediction / scale, actions / scale))
        gradient = torch.autograd.grad(losses["qpos"], captured[0], retain_graph=True)[0].reshape_as(actions)
        torch.testing.assert_close(gradient, 2 * (prediction - actions) / (scale.square() * actions.numel()))
        torch.testing.assert_close(losses["loss"], losses["qpos"] + losses["kl"])
        losses["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)

    def test_conditional_prior_trains_with_policy(self):
        torch.manual_seed(0)
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet", vision_encoder_feature_dim=32, cvae_latent_dim=8, cvae_learned_prior=True
        )
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="floating")
        policy = GraspPolicy(cfg, 28, robot)
        posterior, prior = [], []
        policy.generator.register_forward_hook(lambda module, inputs, output: posterior.append(output[1:3]))
        policy.generator.prior_encoder.register_forward_hook(lambda module, inputs, output: prior.append(output))
        losses = policy({"pointcloud": torch.randn(2, 32, 3)}, torch.randn(2, 4, 3, 28))
        mean, log_var = posterior[0]
        prior_mean, prior_log_var = prior[0].chunk(2, dim=-1)
        expected = (
            torch.distributions.kl_divergence(
                torch.distributions.Normal(mean, (0.5 * log_var).exp()),
                torch.distributions.Normal(prior_mean, (0.5 * prior_log_var.clamp(-10, 10)).exp()),
            )
            .sum(-1)
            .mean()
        )
        torch.testing.assert_close(losses["kl"], expected * cfg.kl_weight)
        assert all(torch.isfinite(value) for value in losses.values()) and losses["kl"] > 0
        losses["loss"].backward()
        assert sum(parameter.grad.abs().sum() for parameter in policy.generator.prior_encoder.parameters()) > 0

    @pytest.mark.hf_assets
    @pytest.mark.parametrize("robot_name", ["arx_x5", "panda_allegro", "agilex_aloha", "yam", "xarm7_sharpa"])
    def test_invalid_decoder_joint_receives_a_recovery_gradient(self, robot_name: str):
        torch.manual_seed(0)
        robot = build_experiment_config(f"{robot_name}_tabletop").robot_config
        helper = GraspRobot.from_config(robot)
        limits = helper.actuated_joint_limits_torch
        valid = helper.init_qpos.clamp(limits[:, 0], limits[:, 1])
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_global_feature=True,
            vision_encoder_feature_dim=8,
            cvae_hidden_dim=8,
            cvae_latent_dim=2,
            enable_kinematic_loss=True,
            qpos_weight=0,
            kl_weight=0,
        )
        model = GraspPolicy(cfg, len(valid), robot).eval()
        head = model.generator.decoder.layers[-1][0]
        joint = int(helper.arm_joint_indices[0])
        index = 2 * len(valid) + joint
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(valid.repeat(3))
            head.bias[index] = limits[joint, 1] + 0.5
            model.generator.decoder.skip_output.weight.zero_()
            model.generator.decoder.skip_output.bias.zero_()
        observations = {"pointcloud": torch.randn(1, 32, 3)}
        targets = valid[None, None, None].expand(1, 2, 3, -1)
        losses = model(observations, targets)
        geometry_gradient = torch.autograd.grad(losses["kinematic"], head.bias, retain_graph=True)[0]
        gradient = torch.autograd.grad(losses["loss"], head.bias)[0]
        assert geometry_gradient[index] == 0 and gradient[index] > 0
        with torch.no_grad():
            head.bias[index] -= gradient[index]
        assert model(observations, targets)["joint_limits"] < losses["joint_limits"]
        with torch.no_grad():
            head.bias.copy_(valid.repeat(3))
        assert model(observations, targets)["joint_limits"] == 0


class TestCenteredPolicy:
    @pytest.mark.parametrize("rot6d", [False, True])
    @pytest.mark.parametrize("coupled", [False, True])
    @pytest.mark.parametrize("hidden", [128, 256])
    def test_centered_policy_translation_equivariance(self, rot6d: bool, coupled: bool, hidden: int):
        torch.manual_seed(0)
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_feature_dim=32,
            center_pointcloud=True,
            pointcloud_scale=10.0,
            convert_rxyz_to_rot6d=rot6d,
            analytic_euler=True,
            coupled_floating_base=coupled,
            cvae_hidden_dim=hidden,
            use_mean=False,
        )
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="floating")
        policy = GraspPolicy(cfg, 31 if rot6d else 28, robot).eval()
        points = torch.randn(2, 32, 3) * 0.03
        offset = torch.tensor([0.13, -0.07, 0.02])
        actions = torch.randn(2, 4, 3, 28) * 0.1
        shifted = actions.clone()
        shifted[..., :3] += offset
        torch.manual_seed(7)
        losses = policy({"pointcloud": points}, actions)
        torch.manual_seed(7)
        shifted_losses = policy({"pointcloud": points + offset}, shifted)
        for key in losses:
            torch.testing.assert_close(losses[key], shifted_losses[key])
        decoded = []
        hook = policy.generator.decoder.register_forward_hook(lambda module, inputs, output: decoded.append(output))
        torch.manual_seed(7)
        with torch.inference_mode():
            output = policy.get_action({"pointcloud": points})
        assert len(decoded) == 1 and output.shape == (2, 1, 3, 28)
        if coupled:
            torch.testing.assert_close(output[..., 0, 3:6], output[..., 1, 3:6])
            torch.testing.assert_close(output[..., 2, :6], output[..., 1, :6])
        hook.remove()
        torch.manual_seed(7)
        with torch.inference_mode():
            shifted_output = policy.get_action({"pointcloud": points + offset})
        torch.testing.assert_close(shifted_output[..., :3], output[..., :3] + offset)
        if rot6d:
            # Nearly parallel decoded 6D axes amplify FP32 centering roundoff to a few microradians.
            torch.testing.assert_close(shifted_output[..., 3:6], output[..., 3:6], atol=3e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(shifted_output[..., 3:6], output[..., 3:6])
        torch.testing.assert_close(shifted_output[..., 6:], output[..., 6:])

    @pytest.mark.parametrize("coupled", [False, True])
    @pytest.mark.parametrize("fingertip", [False, True])
    def test_centered_prior_penetration_uses_world_coordinates(self, coupled: bool, fingertip: bool):
        torch.manual_seed(0)
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_feature_dim=32,
            center_pointcloud=True,
            enable_penetration_loss=True,
            coupled_floating_base=coupled,
            prior_fingertip_weight=10.0 if fingertip else 0.0,
        )
        robot = RobotConfig(
            name="test", urdf_or_mjcf_path="unused.urdf", base_type="floating", fingertip_link_names=["tip"]
        )
        policy = GraspPolicy(cfg, 28, robot)
        policy.robot_helper = SimpleNamespace(
            device=torch.device("cpu"),
            link_names=["tip"],
            canonical_collision_spheres=(torch.tensor([[0.0, 0.0, 0.0, 2.0]]), torch.tensor([0])),
            forward_kinematics_torch=lambda q: q,
            get_collision_spheres=lambda link_poses: torch.cat(
                [link_poses[..., :3], torch.full_like(link_poses[..., :1], 2.0)], dim=-1
            ).unsqueeze(-2),
        )
        points = torch.randn(2, 32, 3) * 0.03
        actions = torch.randn(2, 4, 3, 28) * 0.1
        losses = []
        tip_losses = []
        for shift in (0.0, 0.4):
            offset = torch.tensor([0.0, 0.0, shift])
            shifted_actions = actions.clone()
            shifted_actions[..., :3] += offset
            policy._scene_geom = SimpleNamespace(
                query_target_object_signed_distances=lambda points, shift=shift: (points[..., 2] - shift, None, None)
            )
            torch.manual_seed(9)
            result = policy({"pointcloud": points + offset}, shifted_actions)
            losses.append(result["penetration"])
            if fingertip:
                tip_losses.append(result["prior_fingertip"])
        assert losses[0] > 0
        torch.testing.assert_close(losses[0], losses[1])
        if fingertip:
            torch.testing.assert_close(tip_losses[0], tip_losses[1])
            torch.testing.assert_close(
                tip_losses[0] / 10, losses[0] / cfg.penetration_weight + cfg.penetration_threshold
            )
        (losses[1] + (tip_losses[1] if fingertip else 0)).backward()
        assert policy.generator.decoder.layers[-1][0].weight.grad.abs().sum() > 0

    def test_training_yaw_rotates_points_and_targets_together(self):
        torch.manual_seed(0)
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_feature_dim=32,
            center_pointcloud=True,
            convert_rxyz_to_rot6d=True,
            analytic_euler=True,
            train_yaw_augmentation=True,
        )
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="floating")
        policy = GraspPolicy(cfg, 31, robot)
        points = torch.randn(2, 32, 3) * 0.1
        actions = torch.randn(2, 4, 3, 28)
        original_points, original_actions = points.clone(), actions.clone()
        seen_points, seen_actions = [], []
        policy.vision_encoder.register_forward_pre_hook(lambda module, inputs: seen_points.append(inputs[0]))
        policy.generator.register_forward_pre_hook(lambda module, inputs: seen_actions.append(inputs[0]))
        torch.manual_seed(8)
        yaw = torch.zeros(2, 3)
        yaw[:, 2] = torch.rand(2) * (2 * torch.pi)
        rotation = euler_angles_to_matrix(yaw, "rxyz")
        torch.manual_seed(8)
        losses = policy({"pointcloud": points}, actions)
        center = points.mean(dim=1, keepdim=True)
        torch.testing.assert_close(seen_points[0], (rotation[:, None] @ (points - center).unsqueeze(-1)).squeeze(-1))
        encoded = seen_actions[0].reshape(2, 4, 3, 31)
        torch.testing.assert_close(
            encoded[..., :3], (rotation[:, None, None] @ (actions[..., :3] - center[:, None]).unsqueeze(-1)).squeeze(-1)
        )
        torch.testing.assert_close(
            rotation_6d_to_matrix(encoded[..., 3:9]),
            rotation[:, None, None] @ euler_angles_to_matrix(actions[..., 3:6], "rxyz"),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(encoded[..., 9:], actions[..., 6:])
        torch.testing.assert_close(points, original_points)
        torch.testing.assert_close(actions, original_actions)
        losses["loss"].backward()
        assert policy.generator.decoder.layers[-1][0].weight.grad.abs().sum() > 0
        with torch.inference_mode():
            output = policy.get_action({"pointcloud": points})
        torch.testing.assert_close(seen_points[1], points - center)
        assert output.shape == (2, 1, 3, 28)

    def test_canonical_yaw_is_equivariant_and_single_pass(self):
        torch.manual_seed(3)
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_feature_dim=32,
            center_pointcloud=True,
            convert_rxyz_to_rot6d=True,
            analytic_euler=True,
            coupled_floating_base=True,
            canonical_yaw=True,
            use_mean=False,
        )
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="floating")
        policy = GraspPolicy(cfg, 31, robot).eval()
        points = torch.randn(2, 128, 3) * torch.tensor([0.1, 0.03, 0.05])
        actions = torch.randn(2, 4, 3, 28) * 0.2
        rotation = euler_angles_to_matrix(torch.tensor([[0.0, 0.0, 2.3], [0.0, 0.0, -1.4]]), "rxyz")
        offset = torch.tensor([0.13, -0.2, 0.04])
        shifted_points = (rotation[:, None] @ points.unsqueeze(-1)).squeeze(-1) + offset
        shifted_actions = actions.clone()
        shifted_actions[..., :3] = (rotation[:, None, None] @ actions[..., :3].unsqueeze(-1)).squeeze(-1) + offset
        shifted_actions[..., 3:6] = matrix_to_nearest_euler_angles(
            rotation[:, None, None] @ euler_angles_to_matrix(actions[..., 3:6], "rxyz")
        )
        torch.manual_seed(11)
        original_losses = policy({"pointcloud": points}, actions)
        torch.manual_seed(11)
        shifted_losses = policy({"pointcloud": shifted_points}, shifted_actions)
        for name in original_losses:
            torch.testing.assert_close(original_losses[name], shifted_losses[name], atol=1e-5, rtol=1e-5)
        calls = []
        hook = policy.generator.decoder.register_forward_hook(lambda module, inputs, output: calls.append(len(output)))
        torch.manual_seed(11)
        with torch.inference_mode():
            original = policy.get_action({"pointcloud": points})
        torch.manual_seed(11)
        with torch.inference_mode():
            shifted = policy.get_action({"pointcloud": shifted_points})
        hook.remove()
        assert calls == [2, 2]
        torch.testing.assert_close(
            shifted[..., :3], (rotation[:, None, None] @ original[..., :3].unsqueeze(-1)).squeeze(-1) + offset
        )
        torch.testing.assert_close(
            euler_angles_to_matrix(shifted[..., 3:6], "rxyz"),
            rotation[:, None, None] @ euler_angles_to_matrix(original[..., 3:6], "rxyz"),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(shifted[..., 6:], original[..., 6:])


@pytest.mark.hf_assets
class TestGuidedNoise:
    @pytest.mark.parametrize(
        "generator,learned_prior", [("cvae", False), ("cvae", True), ("flow", False), ("flow_transformer", False)]
    )
    @pytest.mark.parametrize("use_mean", [False, True])
    @pytest.mark.parametrize("temperature", [0.0, 0.7])
    def test_guided_noise_preserves_sampling_and_scene_batches(
        self, generator: str, learned_prior: bool, use_mean: bool, temperature: float
    ):
        torch.manual_seed(0)
        robot = build_robot_config("arx_x5_arm")
        action_dim = GraspRobot.from_config(robot).num_actuated_joints
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_feature_dim=8,
            generative_model=generator,
            cvae_hidden_dim=8,
            cvae_latent_dim=4,
            cvae_learned_prior=learned_prior,
            cvae_temperature=temperature,
            use_mean=use_mean,
            grasp_evaluator_guidance_steps=8,
        )
        policy = GraspPolicy(cfg, action_dim, robot).eval().requires_grad_(False)
        points = torch.randn(2, 16, 3)
        shape = (128, policy.generator.latent_dim)
        torch.manual_seed(7)
        noise = torch.randn(shape)
        sampled_state = torch.get_rng_state()
        torch.manual_seed(7)
        original_state = torch.get_rng_state()
        with torch.inference_mode():
            expected = policy.get_action({"pointcloud": points}, 64)
        assert torch.equal(torch.get_rng_state(), original_state if use_mean else sampled_state)
        explicit_state = torch.get_rng_state()
        with torch.inference_mode():
            actual = policy.get_action({"pointcloud": points}, 64, noise=noise)
            serial = torch.cat(
                [
                    policy.get_action({"pointcloud": points[i : i + 1]}, 64, noise=noise[i * 64 : (i + 1) * 64])
                    for i in range(2)
                ]
            )
        assert torch.equal(torch.get_rng_state(), explicit_state)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual, serial, atol=1e-5, rtol=1e-4)
        assert actual.shape == (2, 64, 3, action_dim) and actual.dtype == torch.float32
        assert torch.isfinite(actual).all()
        if learned_prior:
            condition = policy.vision_encoder(points[:1]).repeat_interleave(64, 0)
            mean, _ = policy.generator.prior(condition)
            assert mean.stride() == (2 * policy.generator.latent_dim, 1)
            torch.manual_seed(11)
            prior_draw = torch.randn_like(mean)
            prior_state = torch.get_rng_state()
            torch.manual_seed(11)
            plain_draw = torch.randn(64, policy.generator.latent_dim)
            assert prior_draw.stride() == plain_draw.stride() == (policy.generator.latent_dim, 1)
            assert torch.equal(prior_draw, plain_draw) and torch.equal(torch.get_rng_state(), prior_state)
        for invalid in (noise[:1], noise.double(), torch.empty_like(noise, device="meta")):
            with pytest.raises(AssertionError):
                policy.get_action({"pointcloud": points}, 64, noise=invalid)
        cfg.grasp_evaluator_guidance_steps = 0
        with pytest.raises(AssertionError):
            policy.get_action({"pointcloud": points}, 64, noise=noise)
