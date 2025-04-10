import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
from torch import nn
import torch.nn.functional as F

from . import pyro as base

class MuaAutoRegression(base.PyroModel):
    def __init__(self, ablations=[], hidden_dims=128, num_channels=1,
                 num_regressors=4, num_stimuli=4):
        super().__init__()
        self._ablations = ablations
        self._num_channels = num_channels
        self._num_regressors = num_regressors
        self._num_stimuli = num_stimuli
        self._data_dim = self._num_stimuli * (self.num_regressors +\
                         self.num_channels)

        self.ar = nn.Linear(self.num_channels, self.num_channels, bias=False)

        if "repetition" not in self.ablations:
            self.register_buffer("repetition_loc", torch.ones(1))
            self.register_buffer("repetition_scale", torch.ones(1))
            self.repetition_q = nn.Sequential(
                nn.Linear(self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, self.num_channels * 2),
            )

        self.register_buffer("x0_loc", torch.zeros(1))
        self.register_buffer("x0_scale", torch.ones(1))
        self.x0_q = nn.Sequential(
            nn.Linear(self._data_dim, hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, self.num_channels * 2)
        )

        if "selectivity" not in self.ablations:
            self.register_buffer("selectivity_p_concentration", torch.ones(2))
            self.register_buffer("selectivity_p_rate", torch.ones(2))
            self.selectivity_q = nn.Sequential(
                nn.Linear(self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, 2 * 2 * self.num_channels), nn.Softplus()
            )

        if "surprise" not in self.ablations:
            self.register_buffer("surprise_p_concentration", torch.ones(1))
            self.register_buffer("surprise_p_rate", torch.ones(1))
            self.surprise_q = nn.Sequential(
                nn.Linear(self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, self.num_channels * 2), nn.Softplus()
            )

    @property
    def ablations(self):
        return self._ablations

    def guide(self, muae, regressors):
        B = muae.shape[0]
        data = torch.cat((muae, self.trial_regressors(regressors)),
                         dim=-1).flatten(-2, -1)
        loc, scale = self.x0_q(data).unbind(dim=-1)
        x0 = pyro.sample("x0", dist.Normal(
            loc.unsqueeze(-1), F.softplus(scale).unsqueeze(-1)
        ).to_event(1))
        P = x0.shape[0]

        if "selectivity" not in self.ablations:
            features = self.selectivity_q(data).expand(P, B, 4) + 1e-5
            concentration, rate = features[:, :, :2], features[:, :, 2:]
            selectivity = pyro.sample("selectivity",
                                      dist.Gamma(concentration,
                                                 rate).to_event(1))

        if "repetition" not in self.ablations:
            features = self.repetition_q(data).expand(P, B, 2)
            loc, scale = features[:, :, 0:1], F.softplus(features[:, :, 1:2])
            pyro.sample("repetition", dist.Normal(loc, scale).to_event(1))

        if "surprise" not in self.ablations:
            features = self.surprise_q(data).expand(P, B, 2) + 1e-5
            concentration, rate = features[:, :, 0:1], features[:, :, 1:2]
            surprise = pyro.sample("surprise",
                                   dist.Gamma(concentration, rate).to_event(1))

    def model(self, muae, regressors):
        # regressors[:, 0:1] = one-hot for orientation (0 -> 45, 1 -> 135)
        # regressors[:, 2] = stimulus repetition count up to current stimulus
        # regressors[:, 3:6] = surprisals
        regressors = self.trial_regressors(regressors)
        B = regressors.shape[0]
        loc, scale = self.x0_loc.expand(B, 1), self.x0_scale.expand(B, 1)
        x = pyro.sample("x0", dist.Normal(loc, scale).to_event(1))
        P = x.shape[0]
        if muae is not None:
            muae = muae.expand(P, B, self._num_stimuli, 1)

        if "repetition" in self.ablations:
            repetition = regressors.new_zeros(torch.Size((P, B, 1)))
        else:
            loc = self.repetition_loc.expand(B, 1)
            scale = self.repetition_scale.expand(B, 1)
            repetition = pyro.sample("repetition",
                                     dist.Normal(loc, scale).to_event(1))

        if "selectivity" in self.ablations:
            selectivity = regressors.new_zeros(torch.Size((P, B, 2)))
        else:
            concentration = self.selectivity_p_concentration.expand(B, 2)
            rate = self.selectivity_p_rate.expand(B, 2)
            selectivity = pyro.sample("selectivity",
                                      dist.Gamma(concentration,
                                                 rate).to_event(1))

        if "surprise" in self.ablations:
            surprise = regressors.new_zeros(torch.Size((P, B, 1)))
        else:
            concentration = self.surprise_p_concentration.expand(B, 1)
            rate = self.surprise_p_rate.expand(B, 1)
            surprise = pyro.sample("surprise",
                                   dist.Gamma(concentration, rate).to_event(1))

        coefficients = torch.cat((selectivity, repetition,
                                  surprise), dim=-1)[:,:,self.regressor_indices]
        regressors = regressors.expand(P, *regressors.shape)
        coefficients = coefficients.unsqueeze(-2).expand(*regressors.shape)
        us = torch.linalg.vecdot(coefficients, regressors).unsqueeze(dim=-1)

        predictions = []
        for p in range(self._num_stimuli):
            x = self.ar(x) + us[:, :, p]
            predictions.append(x)
            x = pyro.sample("x_%d" % p, dist.Normal(x, 0.1).to_event(1),
                            obs=muae[:, :, p] if muae is not None else None)

        return torch.stack(predictions, dim=-2)

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def num_regressors(self):
        return len(self.regressor_indices)

    @property
    def regressor_indices(self):
        indices = []
        if "selectivity" not in self.ablations:
            indices += [0, 1]
        if "repetition" not in self.ablations:
            indices.append(2)
        if "surprise" not in self.ablations:
            indices.append(3)
        return indices

    def trial_regressors(self, regressors):
        return regressors[:, :, self.regressor_indices]
