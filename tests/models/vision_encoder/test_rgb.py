"""Tests for `graspkit.models.vision_encoder.rgb`."""

import pytest
import torch

from graspkit.models.vision_encoder.rgb import build_rgb_encoder, preprocess_rgb


class TestRgbEncoder:
    @pytest.mark.parametrize("name", ["rgb", "rgb_resnet", "rgb_resnet_views"])
    def test_rgb_colour_normalization_and_view_order(self, name: str):
        views = 4 if name == "rgb_resnet_views" else 3
        white = torch.arange(2 * views).reshape(2, views) % 2
        images = (white[:, :, None, None, None] * 255).expand(2, views, 3, 17, 23).flatten(1, 2).to(torch.uint8)
        actual = preprocess_rgb(images, name)
        if name == "rgb":
            expected = white.repeat_interleave(3, dim=1).float()
        else:
            black_colour = torch.tensor([-2.11790393, -2.03571428, -1.80444444])
            white_colour = torch.tensor([2.24890830, 2.42857143, 2.64])
            expected = torch.where(white[:, :, None].bool(), white_colour, black_colour)
            expected = expected.reshape(2 * views, 3) if name == "rgb_resnet_views" else expected.flatten(1)
        assert actual.dtype == torch.float32
        assert actual.shape == (*expected.shape, 128, 128)
        torch.testing.assert_close(actual, expected[:, :, None, None].expand_as(actual), atol=5e-7, rtol=0)

    @pytest.mark.parametrize("name", ["rgb", "rgb_resnet", "rgb_resnet_views"])
    def test_rgb_encoder_features_gradients_and_checkpoint_layout(self, name: str):
        if name != "rgb":
            pytest.importorskip("torchvision")
        encoder = build_rgb_encoder(name, feature_dim=16).eval()
        channels = 12 if name == "rgb_resnet_views" else 9
        images = torch.full((2, channels, 17, 23), 128.0, requires_grad=True)
        features = encoder(preprocess_rgb(images, name))
        assert features.shape == (8 if name == "rgb_resnet_views" else 2, 16)
        features.square().sum().backward()
        assert torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0
        state = encoder.state_dict()
        if name == "rgb":
            assert state["0.weight"].shape == (32, 9, 5, 5)
            assert state["10.weight"].shape == (16, 128 * 8 * 8)
        else:
            assert state["conv1.weight"].shape == (64, 3 if name == "rgb_resnet_views" else 9, 7, 7)
            assert state["fc.weight"].shape == (16, 512 * 4 * 4)
            assert encoder.avgpool.output_size == (4, 4)
