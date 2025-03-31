import monotonicnetworks as lmn
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
from torch import nn
import torch.nn.functional as F

from . import pyro as base

class MultiunitActivityRnn(base.PyroModel):
    def __init__(self, ablations=[], hidden_dims=128, num_regressors=4,
                 num_stimuli=4, state_dims=10):
        super().__init__()
        self._ablations = ablations
        self._num_regressors = num_regressors
        self._num_stimuli = num_stimuli
        self._state_dims = state_dims
        self._data_dim = num_regressors - len(self.ablations) + 1

        if "repetition" not in self.ablations:
            self.register_buffer("adaptation_concentration", torch.ones(1))
            self.register_buffer("adaptation_rate", torch.ones(1))
            self.adaptation_q = nn.Sequential(
                nn.Linear(num_stimuli * self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, 2), nn.Softplus()
            )

        self.decoder = nn.Sequential(
            nn.Linear(state_dims, hidden_dims), nn.SiLU(),
            nn.Linear(hidden_dims, 1),
        )
        monotonicity = []
        if "selectivity" not in self.ablations:
            monotonicity = monotonicity + [1, 1]
        if "repetition" not in self.ablations:
            monotonicity.append(-1)
        if "surprise" not in self.ablations:
            monotonicity.append(1)
        monotonicity.append(0)
        self.dynamics = nn.RNNCell(self._data_dim, state_dims)
        self.register_buffer("h_init_loc", torch.zeros(state_dims))
        self.register_buffer("h_init_scale", torch.ones(state_dims))
        self.h_init_q = nn.Sequential(
            nn.Linear(num_stimuli * self._data_dim, hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, 2 * state_dims)
        )

        self.register_buffer("mixture_p_alpha", torch.ones(2))
        self.mixture_q = nn.Sequential(
            nn.Linear(num_stimuli * self._data_dim, hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, 2), nn.Softplus()
        )

        if "selectivity" not in self.ablations:
            self.register_buffer("angle_alpha", torch.ones(2))
            self.register_buffer("selectivity_concentration", torch.ones(1))
            self.register_buffer("selectivity_rate", torch.ones(1))
            self.selectivity_q = nn.Sequential(
                nn.Linear(num_stimuli * self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, 4), nn.Softplus()
            )

        if "surprise" not in self.ablations:
            self.register_buffer("surprise_concentration", torch.ones(1))
            self.register_buffer("surprise_rate", torch.ones(1))
            self.surprise_q = nn.Sequential(
                nn.Linear(num_stimuli * self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, 2), nn.Softplus()
            )

    @property
    def ablations(self):
        return self._ablations

    def guide(self, muae, regressors):
        B = muae.shape[0]
        data = torch.cat((muae, regressors[:, :, self.regressor_indices]),
                         dim=-1).flatten(-2, -1)
        loc, log_scale = self.h_init_q(data).view(
            -1, self._state_dims, 2
        ).unbind(dim=-1)
        h = pyro.sample("h0", dist.Normal(loc, log_scale.exp()).to_event(1))
        P = h.shape[0]

    def model(self, muae, regressors):
        # regressors[:, 0:1] = one-hot for orientation (0 -> 45, 1 -> 135)
        # regressors[:, 2] = stimulus repetition count up to current stimulus
        # regressors[:, 3:6] = surprisals
        regressors = regressors[:, :, self.regressor_indices]
        B = regressors.shape[0]
        loc = self.h_init_loc.expand(B, -1)
        scale = self.h_init_scale.expand(B, -1)
        h = pyro.sample("h0", dist.Normal(loc, scale).to_event(1))
        P = h.shape[0]
        if muae is not None:
            muae = muae.expand(P, B, self._num_stimuli, 1)
        regressors = regressors.expand(P, B, self._num_stimuli,
                                       self.num_regressors)

        predictions = []
        x = regressors.new_zeros(torch.Size((P, B, 1)))
        for p in range(self._num_stimuli):
            us = torch.cat((regressors[:, :, p], x), dim=-1)
            h = self.dynamics(us.flatten(0, 1), h.flatten(0, 1)).view(P, B, -1)
            x = self.decoder(h)
            predictions.append(x)
            x = pyro.sample("x_%d" % p, dist.Normal(x, 0.1).to_event(1),
                            obs=muae[:, :, p] if muae is not None else None)

        return torch.stack(predictions, dim=-2)

    @property
    def num_regressors(self):
        return self._num_regressors - len(self.ablations)

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
