from typing import Literal

import torch
from torch import nn

from graspkit.models.mlp import MLP


class FlowMatching(nn.Module):
    """Linear-path flow matching with an MLP or transformer velocity network."""

    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        hidden_dims: list[int],
        model: Literal["flow", "flow_transformer"] = "flow",
        num_queries: int = 3,
    ):
        super().__init__()
        assert model in ("flow", "flow_transformer")
        self.model = model
        self.latent_dim = x_dim
        self.learned_prior = False
        self.register_buffer("action_center", torch.zeros(x_dim))
        self.register_buffer("action_scale", torch.ones(x_dim))
        if model == "flow":
            self.decoder = MLP(input_dim=x_dim + cond_dim + 1, output_dim=x_dim, hidden_dims=hidden_dims)
        else:
            self.num_queries = num_queries
            width = hidden_dims[0]
            self.x_proj = nn.Linear(x_dim // num_queries, width)
            self.cond_proj = nn.Linear(cond_dim, width)
            self.t_embed = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
            self.action_tokens = nn.Parameter(torch.randn(num_queries, width) * 0.02)
            self.decoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    width, 8, width * 4, dropout=0, activation=nn.functional.silu, batch_first=True
                ),
                num_layers=len(hidden_dims),
                enable_nested_tensor=False,
            )
            self.action_head = nn.Linear(width, x_dim // num_queries)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return loss reconstruction, zero latent moments, and timestep-correct geometry."""
        x = (x - self.action_center) / self.action_scale
        noise = torch.randn_like(x)
        time = torch.rand(len(x), 1, device=x.device, dtype=x.dtype)
        interpolant = (1 - time) * noise + time * x
        velocity = self.velocity(interpolant, cond, time)
        return (
            self.action_center + self.action_scale * (noise + velocity),
            torch.zeros_like(x),
            torch.zeros_like(x),
            self.action_center + self.action_scale * (interpolant + (1 - time) * velocity),
        )

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for step in range(8):
            time = z.new_full((len(z), 1), step / 8)
            z = z + self.velocity(z, cond, time) / 8
        return self.action_center + self.action_scale * z

    def velocity(self, x: torch.Tensor, cond: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if self.model == "flow":
            return self.decoder(torch.cat([x, cond, time], dim=-1))
        tokens = self.x_proj(x.reshape(len(x), self.num_queries, -1)) + self.action_tokens
        tokens = tokens + self.t_embed(time)[:, None]
        memory = self.cond_proj(cond)[:, None]
        return self.action_head(self.decoder(torch.cat([tokens, memory], dim=1))[:, :-1]).flatten(1)

    def sample(self, cond: torch.Tensor, use_mean: bool, temperature: float = 1.0) -> torch.Tensor:
        """Integrate from zero or a temperature-scaled standard-normal draw."""
        z = cond.new_zeros(cond.shape[0], self.latent_dim)
        return self.decode(z if use_mean else torch.randn_like(z) * temperature, cond)
