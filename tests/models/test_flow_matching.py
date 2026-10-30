"""Tests for `graspkit.models.flow_matching`."""

import pytest
import torch

from graspkit.models.flow_matching import FlowMatching


class TestFlowMatching:
    @pytest.mark.parametrize("model", ["flow", "flow_transformer"])
    @pytest.mark.parametrize("normalized", [False, True])
    def test_flow_interpolant_and_integration(self, model: str, normalized: bool):
        generator = FlowMatching(6, 4, [16], model=model, num_queries=3)
        with torch.no_grad():
            for parameter in generator.parameters():
                parameter.zero_()
            if model == "flow":
                generator.decoder.layers[-1][0].bias.fill_(0.5)
            else:
                generator.action_head.bias.fill_(0.5)
            if normalized:
                generator.action_center.copy_(torch.arange(6) * 0.1)
                generator.action_scale.copy_(torch.tensor([0.01, 0.02, 0.1, 1.0, 2.0, 3.0]))
        condition = torch.zeros(2, 4)
        origin = torch.randn(2, 6, requires_grad=True)
        decoded = generator.decode(origin, condition)
        torch.testing.assert_close(decoded, generator.action_center + generator.action_scale * (origin + 0.5))
        torch.testing.assert_close(
            torch.autograd.grad(decoded.sum(), origin)[0], generator.action_scale.expand_as(origin)
        )
        target = torch.randn(2, 6)
        torch.manual_seed(7)
        noise = torch.randn_like(target)
        time = torch.rand(len(target), 1)
        torch.manual_seed(7)
        reconstructed, mean, log_var, geometry = generator(target, condition)
        torch.testing.assert_close(reconstructed, generator.action_center + generator.action_scale * (noise + 0.5))
        torch.testing.assert_close(
            ((reconstructed - target) / generator.action_scale).square(),
            (0.5 - ((target - generator.action_center) / generator.action_scale - noise)).square(),
        )
        assert not mean.any() and not log_var.any()
        torch.testing.assert_close(geometry, time * target + (1 - time) * reconstructed)
        torch.testing.assert_close(
            (geometry - target) / generator.action_scale,
            (1 - time) * (reconstructed - target) / generator.action_scale,
        )
