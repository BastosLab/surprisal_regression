import monotonicnetworks as lmn
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
from torch import nn
import torch.nn.functional as F

from . import pyro as base

class MonotonicallyDrivenRnnCell(pnn.PyroModule):
    def __init__(self, input_dims, hidden_dims, constraints):
        super().__init__()
        assert constraints.shape == (input_dims, hidden_dims)
        self.drive = lmn.MonotonicLayer(
            input_dims, hidden_dims,
            monotonic_constraints=constraints.to(dtype=torch.float)
        )
        self.dynamics = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, input, h):
        return F.tanh(self.drive(input) + self.dynamics(h))

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

        self.decoder = lmn.MonotonicLayer(
            state_dims, 1,
            monotonic_constraints=torch.tensor([[1.] * state_dims]).T
        )
        monotonicity = []
        if "selectivity" not in self.ablations:
            monotonicity = monotonicity + [1, 1]
        if "repetition" not in self.ablations:
            monotonicity.append(-1)
        if "surprise" not in self.ablations:
            monotonicity.append(1)
        monotonicity.append(0)
        self.dynamics = MonotonicallyDrivenRnnCell(
            self._data_dim, state_dims,
            torch.tensor([monotonicity] * state_dims).T
        )
        self.register_buffer("h_init_loc", torch.zeros(state_dims))
        self.register_buffer("h_init_scale", torch.ones(state_dims))
        self.h_init_q = nn.Sequential(
            nn.Linear(num_stimuli * self._data_dim, hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, 2 * state_dims)
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

        if "repetition" not in self.ablations:
            features = self.adaptation_q(data).expand(P, B, 2) + 1e-5
            concentration, rate = features[:, :, 0:1], features[:, :, 1:2]
            pyro.sample("adaptation",
                        dist.Gamma(concentration, rate).to_event(1))

        if "selectivity" not in self.ablations:
            features = self.selectivity_q(data).expand(P, B, 4)
            alpha = features[:, :, :2] + 1e-5
            pyro.sample("orientation", dist.Dirichlet(alpha))

            concentration, rate = features[:, :, 2:3], features[:, :, 3:4]
            pyro.sample("selectivity",
                        dist.Gamma(concentration, rate).to_event(1))

        if "surprise" not in self.ablations:
            features = self.surprise_q(data).expand(P, B, 2)
            concentration, rate = features[:, :, 0:1], features[:, :, 1:2]
            pyro.sample("surprise", dist.Gamma(concentration, rate).to_event(1))

    def model(self, muae, regressors):
        # regressors[:, 0:1] = one-hot for orientation (0 -> 45, 1 -> 135)
        # regressors[:, 2] = stimulus repetition count up to current stimulus
        # regressors[:, 3:6] = surprisals
        B = regressors.shape[0]
        loc = self.h_init_loc.expand(B, -1)
        scale = self.h_init_scale.expand(B, -1)
        h = pyro.sample("h0", dist.Normal(loc, scale).to_event(1))
        P = h.shape[0]
        if muae is not None:
            muae = muae.expand(P, B, self._num_stimuli, 1)

        if "selectivity" in self.ablations:
            selectivity = regressors.new_zeros(torch.Size((P, B, 2)))
        else:
            alpha = self.angle_alpha.expand(B, 2)
            selectivity = pyro.sample("orientation", dist.Dirichlet(alpha))
            concentration = self.selectivity_concentration.expand(B, 1)
            rate = self.selectivity_rate.expand(B, 1)
            selectivity = selectivity * pyro.sample(
                "selectivity", dist.Gamma(concentration, rate).to_event(1)
            )

        if "repetition" in self.ablations:
            adaptation = regressors.new_zeros(torch.Size((P, B, 1)))
        else:
            concentration = self.adaptation_concentration.expand(B, 1)
            rate = self.adaptation_rate.expand(B, 1)
            adaptation = pyro.sample(
                "adaptation", dist.Gamma(concentration, rate).to_event(1)
            )

        if "surprise" in self.ablations:
            surprise = regressors.new_zeros(torch.Size((P, B, 1)))
        else:
            concentration = self.surprise_concentration.expand(B, 1)
            rate = self.surprise_rate.expand(B, 1)
            surprise = pyro.sample("surprise",
                                   dist.Gamma(concentration, rate).to_event(1))

        coefficients = torch.cat((selectivity, -adaptation, surprise), dim=-1)
        regressors = regressors.expand(P, *regressors.shape)
        regressions = coefficients.unsqueeze(dim=-2) * regressors

        predictions = []
        x = regressors.new_zeros(torch.Size((P, B, 1)))
        for p in range(self._num_stimuli):
            us = torch.cat((regressions[:, :, p, self.regressor_indices], x),
                           dim=-1)
            h = self.dynamics(us.flatten(0, 1), h.flatten(0, 1)).view(P, B, -1)
            x = self.decoder(h)
            predictions.append(x)
            x = pyro.sample("x_%d" % p, dist.Normal(x, 0.1).to_event(1),
                            obs=muae[:, :, p] if muae is not None else None)

        return torch.stack(predictions, dim=-2)

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
