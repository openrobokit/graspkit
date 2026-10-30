"""SparseUNet driven by SpConv, adapted from https://github.com/Pointcept/Pointcept.

This module requires the installation of the following packages:

- spconv: https://github.com/traveller59/spconv?tab=readme-ov-file#spconv-spatially-sparse-convolution-library

Original Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com); please cite their work if used in research.
"""

import logging
import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Literal, Optional

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.utils import scatter


class VoxelDropout(nn.Module):
    """Random voxel dropout for data augmentation, like PatchDropout in point_encoder.py but for sparse voxel data."""

    def __init__(self, prob, exclude_cls_token=False):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_cls_token = exclude_cls_token
        logging.info("voxel dropout prob is {}".format(prob))

    def forward(self, sparse_tensor):
        if not self.training or self.prob == 0.0:
            return sparse_tensor

        # Get features and indices from sparse tensor
        features = sparse_tensor.features
        indices = sparse_tensor.indices
        spatial_shape = sparse_tensor.spatial_shape
        batch_size = sparse_tensor.batch_size

        # Create dropout mask
        keep_prob = 1 - self.prob
        num_voxels = features.size(0)
        num_voxels_keep = max(1, int(num_voxels * keep_prob))

        # Randomly select voxels to keep
        rand = torch.randn(num_voxels, device=features.device)
        voxel_indices_keep = rand.topk(num_voxels_keep, dim=-1).indices

        # Apply dropout
        features_kept = features[voxel_indices_keep]
        indices_kept = indices[voxel_indices_keep]

        # Create new sparse tensor with dropped voxels
        sparse_tensor_dropped = spconv.SparseConvTensor(
            features=features_kept,
            indices=indices_kept,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )

        return sparse_tensor_dropped


# Adapted from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py
def _trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Generate via a truncated uniform distribution, then apply the inverse CDF for the normal distribution.
    l = norm_cdf((a - mean) / std)  # noqa: E741
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated standard normal.
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills tensor from N(mean,std), truncated to [a,b] post-sample (unlike torch's trunc_normal_; adjust a,b).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


# Adapted from: https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/utils/misc.py
@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


# Adapted from: https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/sparse_unet/spconv_unet_v1m1_base.py
class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SpUNetBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        grid_size: float = 0.005,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=False,
        normalization: Optional[Literal["layernorm", "groupnorm"]] = None,
        num_groups: int = 32,
        voxel_dropout_prob: float = 0.0,
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.grid_size = grid_size
        self.cls_mode = cls_mode

        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)  # # NOTE BN causes error for small batch size
        if normalization == "layernorm":
            norm_fn = partial(nn.LayerNorm, eps=1e-4)
        elif normalization == "groupnorm":
            norm_fn = partial(nn.GroupNorm, num_groups=num_groups, eps=1e-4)
        elif normalization is None:
            norm_fn = partial(nn.Identity)
        else:
            raise ValueError(f"Normalization {normalization} not supported")

        # Add voxel dropout
        self.voxel_dropout = VoxelDropout(voxel_dropout_prob) if voxel_dropout_prob > 0.0 else nn.Identity()

        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(inplace=True),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.cls_mode else None

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels, channels[s], kernel_size=2, stride=2, bias=False, indice_key=f"spconv{s + 1}"
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(inplace=True),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                block(channels[s], channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            if not self.cls_mode:
                # decode num_stages
                self.up.append(
                    spconv.SparseSequential(
                        spconv.SparseInverseConv3d(
                            channels[len(channels) - s - 2],
                            dec_channels,
                            kernel_size=2,
                            bias=False,
                            indice_key=f"spconv{s + 1}",
                        ),
                        norm_fn(dec_channels),
                        nn.ReLU(inplace=True),
                    )
                )
                self.dec.append(
                    spconv.SparseSequential(
                        OrderedDict(
                            [
                                (
                                    (
                                        f"block{i}",
                                        block(
                                            dec_channels + enc_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                    if i == 0
                                    else (
                                        f"block{i}",
                                        block(dec_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"),
                                    )
                                )
                                for i in range(layers[len(channels) - s - 1])
                            ]
                        )
                    )
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        final_in_channels = channels[-1] if not self.cls_mode else channels[self.num_stages - 1]
        self.final = (
            spconv.SubMConv3d(final_in_channels, num_classes, kernel_size=1, padding=1, bias=True)
            if num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, points: torch.Tensor):
        B, N, D = points.shape
        # TODO: Check if this is the correct way to normalize the points
        grid_coord = torch.div(
            points[..., :3] - points[..., :3].min(-2, keepdim=True).values, self.grid_size, rounding_mode="trunc"
        ).int()
        feat = rearrange(points, "B N D -> (B N) D")
        grid_coord = rearrange(grid_coord, "b n d -> (b n) d")
        batch = torch.arange(B, device=points.device, dtype=torch.int).repeat_interleave(N)

        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1).int(), grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv_input(x)
        x = self.voxel_dropout(x)  # Apply voxel dropout after initial convolution
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        if not self.cls_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)

        x = self.final(x)
        if self.cls_mode:
            x = x.replace_feature(scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0))
            return x.features
        else:
            return x.features.view(B, N, -1)


class SpUNetCls(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ):
        super().__init__()
        self.model = SpUNetBase(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            channels=channels,
            layers=layers,
            cls_mode=True,
        )

    def forward(self, points: torch.Tensor):
        x = self.model(points)
        return F.log_softmax(x, dim=1)


class SpUNetSeg(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ):
        super().__init__()
        self.model = SpUNetBase(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            channels=channels,
            layers=layers,
            cls_mode=False,
        )

    def forward(self, points: torch.Tensor):
        x = self.model(points)
        x = x.view(points.shape[:-1] + (x.shape[-1],))
        return F.log_softmax(x, dim=-1)


class SpUNetEnc(nn.Module):
    def __init__(
        self,
        in_channels: int,
        enc_dim: int,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        grid_size: float = 0.005,
        normalization: Optional[Literal["layernorm", "groupnorm"]] = None,
        voxel_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.model = SpUNetBase(
            in_channels=in_channels,
            num_classes=enc_dim,  # Output of base model will be `enc_dim`
            base_channels=base_channels,
            channels=channels,
            layers=layers,
            cls_mode=True,
            grid_size=grid_size,
            normalization=normalization,
            voxel_dropout_prob=voxel_dropout_prob,
        )
        self.linear = nn.Linear(enc_dim, enc_dim)
        self.norm = nn.LayerNorm(enc_dim)

    def forward(self, points: torch.Tensor):
        x = self.model(points)
        x = self.linear(x)
        x = self.norm(x)
        return x
