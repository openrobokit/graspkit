"""PointNet in PyTorch, adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/."""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float


class PointNetEncoder(nn.Module):
    """PointNet encoder from upstream `pointnet_utils.py` without spatial transformers or BN (breaks small batches)."""

    def __init__(self, input_dim: int = 3, feature_dim: int = 1024, global_feature: bool = True):
        """Initialize PointNet encoder.

        Args:
            input_dim (int, optional): Dimension of input points. Defaults to 3.
            feature_dim (int, optional): Dimension of extracted features. Defaults to 1024.
            global_feature (bool, optional): If True, return global feature; else per-point features. Defaults to True.
        """
        super().__init__()
        self.global_feat = global_feature
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.relu = nn.ReLU(inplace=True)
        if feature_dim != 1024:
            self.feat_linear: Optional[nn.Linear] = torch.nn.Linear(1024 if global_feature else 1024 + 64, feature_dim)
        else:
            self.feat_linear = None

    def forward(
        self, x: Float[torch.Tensor, "batch num_points dim"]
    ) -> Float[torch.Tensor, "batch feature_dim"] | Float[torch.Tensor, "batch num_points feature_dim"]:
        x = rearrange(x, "batch num_points dim -> batch dim num_points")
        num_points = x.shape[-1]
        x = self.relu(self.conv1(x))
        pointfeat = x
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            if self.feat_linear is not None:
                x = self.feat_linear(x)
            return x
        else:
            x = repeat(x, "b d -> b d n", n=num_points)
            x = torch.cat([x, pointfeat], 1)
            if self.feat_linear is not None:
                x = self.feat_linear(rearrange(x, "b d n -> b n d"))
            return x
