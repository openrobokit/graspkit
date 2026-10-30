from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A flexible MLP: SiLU + LayerNorm hidden blocks, residual output connection."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            block = [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:  # only add normalization/activation on hidden layers
                block.append(nn.LayerNorm(dims[i + 1], eps=1e-4))
                block.append(nn.SiLU(inplace=True))
            layers.append(nn.Sequential(*block))
        self.layers = nn.ModuleList(layers)
        self.skip_output = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        for layer in self.layers:
            x = layer(x)
        return self.skip_output(x_in) + x
