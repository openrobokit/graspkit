"""RGB encoder construction and shared image preprocessing."""

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def build_rgb_encoder(name: str, feature_dim: int) -> nn.Module:
    """Return the original module directly, preserving checkpoint keys and feature hooks."""
    if name == "rgb":
        return nn.Sequential(
            nn.Conv2d(9, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, feature_dim),
        )
    assert name in ("rgb_resnet", "rgb_resnet_views")
    from torchvision.models import resnet18
    from torchvision.ops.misc import FrozenBatchNorm2d

    # Fixed (ImageNet) normalization statistics: batch statistics of the synthetic views made fine-tuning diverge.
    encoder = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
    if name == "rgb_resnet":
        encoder.conv1 = nn.Conv2d(9, 64, 7, stride=2, padding=3, bias=False)
    encoder.avgpool = nn.AdaptiveAvgPool2d((4, 4))
    encoder.fc = nn.Linear(512 * 4 * 4, feature_dim)
    return encoder


def preprocess_rgb(rgb: torch.Tensor, name: str) -> torch.Tensor:
    """Resize packed RGB views and apply the encoder's existing normalization."""
    rgb = F.interpolate(rgb.float(), size=(128, 128), mode="bilinear", align_corners=False) / 255
    if name in ("rgb_resnet", "rgb_resnet_views"):
        if name == "rgb_resnet_views":
            rgb = rearrange(rgb, "b (v c) h w -> (b v) c h w", c=3)
        rgb = (rgb - rgb.new_tensor([0.485, 0.456, 0.406] * (rgb.shape[1] // 3))[None, :, None, None]) / (
            rgb.new_tensor([0.229, 0.224, 0.225] * (rgb.shape[1] // 3))[None, :, None, None]
        )
    return rgb
