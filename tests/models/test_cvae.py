"""Tests for `graspkit.models.cvae`."""

import pytest
import torch

from graspkit.models.cvae import CVAE


class TestCVAE:
    def test_cvae_normalization_preserves_posterior_and_physical_decoder(self):
        raw = CVAE(6, 4, 3, [8])
        normalized = CVAE(6, 4, 3, [8], normalize_actions=True)
        assert not {"action_center", "action_scale"} & raw.state_dict().keys()
        center = torch.arange(6).float()
        scale = torch.tensor([0.001, 0.01, 0.1, 1.0, 2.0, 3.0])
        normalized.load_state_dict({**raw.state_dict(), "action_center": center, "action_scale": scale}, strict=True)
        target, condition = torch.randn(2, 6), torch.randn(2, 4)
        torch.manual_seed(4)
        original = raw((target - center) / scale, condition)
        torch.manual_seed(4)
        actual = normalized(target, condition)
        torch.testing.assert_close(actual[0], center + scale * original[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], original[1], atol=0, rtol=0)
        torch.testing.assert_close(actual[2], original[2], atol=0, rtol=0)
        torch.testing.assert_close(actual[3], actual[0], atol=0, rtol=0)

    @pytest.mark.parametrize("posterior_log_variance", [-100.0, 100.0])
    def test_extreme_posterior_has_finite_kl_and_gradients(self, posterior_log_variance: float):
        model = CVAE(6, 4, 3, [8])
        model.encoder = torch.nn.Linear(10, 6)
        with torch.no_grad():
            model.encoder.weight.zero_()
            model.encoder.bias.zero_()
            model.encoder.bias[3:] = posterior_log_variance
        reconstructed, means, log_variance, _ = model(torch.zeros(2, 6), torch.zeros(2, 4))
        kl = -0.5 * (1 + log_variance - means.square() - log_variance.exp()).sum(-1).mean()
        loss = reconstructed.square().mean() + kl
        assert torch.isfinite(loss)
        assert log_variance.abs().max() <= 10
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    @pytest.mark.parametrize("use_mean", [False, True])
    def test_conditional_prior_single_draw(self, use_mean: bool):
        torch.manual_seed(0)
        model = CVAE(6, 4, 3, [8], learned_prior=True).eval()
        cond = torch.randn(5, 4)
        mean, log_var = model.prior(cond)
        torch.manual_seed(8)
        z = mean if use_mean else mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        expected_rng = torch.get_rng_state()
        expected = model.decode(z, cond)
        decoded = []
        hook = model.decoder.register_forward_hook(lambda module, inputs, output: decoded.append(output))
        torch.manual_seed(8)
        with torch.inference_mode():
            actual = model.sample(cond, use_mean)
        hook.remove()
        assert len(decoded) == 1 and torch.equal(torch.get_rng_state(), expected_rng)
        torch.testing.assert_close(actual, expected)

    def test_cvae_sampling_temperature(self):
        torch.manual_seed(0)
        model = CVAE(x_dim=6, cond_dim=4, latent_dim=3, hidden_dims=[8])
        cond = torch.randn(2, 4)
        torch.manual_seed(7)
        latent = torch.randn(2, 3)
        torch.manual_seed(7)
        torch.testing.assert_close(model.sample(cond, use_mean=False), model.decode(latent, cond), atol=0, rtol=0)
        torch.manual_seed(7)
        torch.testing.assert_close(
            model.sample(cond, use_mean=False, temperature=1.5), model.decode(latent * 1.5, cond), atol=0, rtol=0
        )
        torch.testing.assert_close(
            model.sample(cond, use_mean=False, temperature=0), model.sample(cond, use_mean=True), atol=0, rtol=0
        )
