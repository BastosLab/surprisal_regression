import pyro
import pyro.distributions as dist
import torch
from torch import nn

from . import pyro as base

class VariationalAutoencoder(base.PyroModel):
    def __init__(self, hidden_dim=400, x_dim=784, z_dim=50):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, x_dim), nn.Sigmoid()
        )

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, z_dim * 2),
        )

        self._x_dim = x_dim
        self._z_dim = z_dim

    def guide(self, xs):
        pyro.module("encoder", self.encoder)
        loc, log_scale = self.encoder(xs.flatten(1, -1)).view(
            xs.shape[0], -1, 2
        ).unbind(dim=-1)
        density = dist.Normal(loc, log_scale.exp()).to_event(1)
        z = pyro.sample("z", density)

        return z

    def model(self, xs):
        pyro.module("decoder", self.decoder)

        loc = xs.new_zeros(torch.Size((xs.shape[0], self._z_dim)))
        scale = xs.new_ones(torch.Size((xs.shape[0], self._z_dim)))
        z = pyro.sample("z", dist.Normal(loc, scale).to_event(1))

        x_estimate = self.decoder(z).view(z.shape[0], *xs.shape)
        pyro.sample("x", dist.ContinuousBernoulli(x_estimate).to_event(3),
                    obs=xs.unsqueeze(dim=0))
        return x_estimate
