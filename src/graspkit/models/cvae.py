import torch
from torch import nn

from graspkit.models.mlp import MLP


class CVAE(nn.Module):
    """Conditional variational autoencoder with an optional learned latent prior."""

    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        learned_prior: bool = False,
        normalize_actions: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.learned_prior = learned_prior
        self.normalize_actions = normalize_actions
        if normalize_actions:
            self.register_buffer("action_center", torch.zeros(x_dim))
            self.register_buffer("action_scale", torch.ones(x_dim))
        self.encoder = MLP(input_dim=x_dim + cond_dim, output_dim=latent_dim * 2, hidden_dims=hidden_dims)
        self.decoder = MLP(input_dim=latent_dim + cond_dim, output_dim=x_dim, hidden_dims=hidden_dims)
        if learned_prior:
            self.prior_encoder = MLP(input_dim=cond_dim, output_dim=latent_dim * 2, hidden_dims=hidden_dims)

    def prior(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means, log_var = self.prior_encoder(cond).chunk(2, dim=-1)
        return means, log_var.clamp(min=-10.0, max=10.0)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return loss reconstruction, latent moments, and timestep-correct geometry reconstruction."""
        if self.normalize_actions:
            x = (x - self.action_center) / self.action_scale
        means, log_var = self.encoder(torch.cat([x, cond], dim=-1)).chunk(2, dim=-1)
        log_var = log_var.clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * log_var)
        z = means + torch.randn_like(std) * std
        reconstruction = self.decode(z, cond)
        return reconstruction, means, log_var, reconstruction

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(torch.cat([z, cond], dim=-1))
        if self.normalize_actions:
            reconstruction = self.action_center + self.action_scale * reconstruction
        return reconstruction

    def sample(self, cond: torch.Tensor, use_mean: bool, temperature: float = 1.0) -> torch.Tensor:
        """Decode the prior mean or a temperature-scaled standard-normal sample."""
        if self.learned_prior:
            means, log_var = self.prior(cond)
            z = means if use_mean else means + torch.randn_like(means) * (0.5 * log_var).exp() * temperature
            return self.decode(z, cond)
        z = cond.new_zeros(cond.shape[0], self.latent_dim)
        return self.decode(z if use_mean else torch.randn_like(z) * temperature, cond)
