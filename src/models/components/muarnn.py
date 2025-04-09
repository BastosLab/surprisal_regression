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
                 num_stimuli=4, state_dims=32):
        super().__init__()
        self._ablations = ablations
        self._num_regressors = num_regressors
        self._num_stimuli = num_stimuli
        self._state_dims = state_dims
        self._data_dim = self.num_regressors + 1

        monotonicity = []
        if "selectivity" not in self.ablations:
            monotonicity = monotonicity + [0., 0.]
        if "repetition" not in self.ablations:
            monotonicity.append(1.)
        if "surprise" not in self.ablations:
            monotonicity.append(1)
        self.decoder = lmn.MonotonicLayer(
            self.num_regressors + state_dims, 1,
            monotonic_constraints=torch.tensor(monotonicity + [0.] * state_dims)
        )
        self.dynamics = nn.GRUCell(self.num_regressors + 1, state_dims)
        self.register_buffer("h_init_loc", torch.zeros(state_dims))
        self.register_buffer("h_init_scale", torch.ones(state_dims))
        self.h_init_q = nn.Sequential(
            nn.Linear(num_stimuli * self._data_dim, hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, 2 * state_dims)
        )

    @property
    def ablations(self):
        return self._ablations

    def guide(self, muae, regressors):
        B = muae.shape[0]
        data = torch.cat((muae, regressors[:, :, self.regressor_indices]),
                         dim=-1).flatten(-2, -1)
        loc, scale = self.h_init_q(data).view(
            -1, self._state_dims, 2
        ).unbind(dim=-1)
        h = pyro.sample("h0", dist.Normal(loc,
                                          F.softplus(scale) + 1e-5).to_event(1))
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
        for p in range(self._num_stimuli):
            x = self.decoder(torch.cat((regressors[:, :, p], h), dim=-1))
            predictions.append(x)
            x = pyro.sample("x_%d" % p, dist.Normal(x, 0.1).to_event(1),
                            obs=muae[:, :, p] if muae is not None else None)

            us = torch.cat((regressors[:, :, p], x), dim=-1)
            h = self.dynamics(us.flatten(0, 1), h.flatten(0, 1)).view(P, B, -1)

        return torch.stack(predictions, dim=-2)

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
