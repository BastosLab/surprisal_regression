import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
from torch import nn

from . import pyro as base

class TrialwiseLinearRegression(base.PyroModel):
    def __init__(self, ablations=[], hidden_dims=128, num_regressors=4,
                 num_stimuli=4):
        super().__init__()
        self._ablations = ablations
        self._num_regressors = num_regressors
        self._num_stimuli = num_stimuli
        self._data_dim = self._num_stimuli * (num_regressors -\
                                              len(self.ablations) + 1)

        if "repetition" not in self.ablations:
            self.register_buffer("adaptation_concentration", torch.ones(1))
            self.register_buffer("adaptation_rate", torch.ones(1))
            self.adaptation_q = nn.Sequential(
                nn.Linear(self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, 2), nn.Softplus()
            )

        self.register_buffer("baseline_p_concentration", torch.ones(1))
        self.register_buffer("baseline_p_rate", torch.ones(1))
        self.baseline_q = nn.Sequential(
            nn.Linear(self._data_dim, hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, 2), nn.Softplus()
        )

        if "selectivity" not in self.ablations:
            self.register_buffer("angle_p_alpha", torch.ones(2))
            self.register_buffer("selectivity_p_concentration", torch.ones(1))
            self.register_buffer("selectivity_p_rate", torch.ones(1))
            self.selectivity_q = nn.Sequential(
                nn.Linear(self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, 4), nn.Softplus()
            )

        if "surprise" not in self.ablations:
            self.register_buffer("surprise_p_concentration", torch.ones(1))
            self.register_buffer("surprise_p_rate", torch.ones(1))
            self.surprise_q = nn.Sequential(
                nn.Linear(self._data_dim, hidden_dims),
                nn.SiLU(),
                nn.Linear(hidden_dims, 2), nn.Softplus()
            )

    @property
    def ablations(self):
        return self._ablations

    def guide(self, muae, regressors):
        B = muae.shape[0]
        data = torch.cat((muae, self.trial_regressors(regressors)),
                         dim=-1).flatten(-2, -1)
        concentration, rate = (self.baseline_q(data) + 1e-5).unbind(dim=-1)
        baseline = pyro.sample("baseline", dist.Gamma(
            concentration.unsqueeze(dim=-1), rate.unsqueeze(dim=-1)
        ).to_event(1))
        P = baseline.shape[0]

        if "selectivity" not in self.ablations:
            features = self.selectivity_q(data).expand(P, B, 4) + 1e-5
            alpha = features[:, :, :2]
            orientation = pyro.sample("orientation", dist.Dirichlet(alpha))

            concentration, rate = features[:, :, 2:3], features[:, :, 3:4]
            selectivity = orientation * pyro.sample(
                "selectivity", dist.Gamma(concentration, rate).to_event(1)
            )

        if "repetition" not in self.ablations:
            features = self.adaptation_q(data).expand(P, B, 2) + 1e-5
            concentration, rate = features[:, :, 0:1], features[:, :, 1:2]
            pyro.sample("adaptation",
                        dist.Gamma(concentration, rate).to_event(1))

        if "surprise" not in self.ablations:
            features = self.surprise_q(data).expand(P, B, 2) + 1e-5
            concentration, rate = features[:, :, 0:1], features[:, :, 1:2]
            surprise = pyro.sample("surprise",
                                   dist.Gamma(concentration, rate).to_event(1))

        # features = self.time_q(data).expand(P, B, 2) + 1e-5
        # concentration, rate = features[:, :, 0:1], features[:, :, 1:2]
        # time = pyro.sample("time", dist.Gamma(concentration, rate).to_event(1))

    def model(self, muae, regressors):
        # regressors[:, 0:1] = one-hot for orientation (0 -> 45, 1 -> 135)
        # regressors[:, 2] = stimulus repetition count up to current stimulus
        # regressors[:, 3:6] = surprisals
        regressors = self.trial_regressors(regressors)
        B = regressors.shape[0]
        concentration = self.baseline_p_concentration.expand(B, 1)
        rate = self.baseline_p_rate.expand(B, 1)
        baseline = pyro.sample("baseline",
                               dist.Gamma(concentration, rate).to_event(1))
        P = baseline.shape[0]

        if "repetition" in self.ablations:
            repetition = regressors.new_zeros(torch.Size((P, B, 1)))
        else:
            concentration = self.adaptation_concentration.expand(B, 1)
            rate = self.adaptation_rate.expand(B, 1)
            adaptation = pyro.sample(
                "adaptation", dist.Gamma(concentration, rate).to_event(1)
            )

        if "selectivity" in self.ablations:
            selectivity = regressors.new_zeros(torch.Size((P, B, 2)))
        else:
            alpha = self.angle_p_alpha.expand(B, 2)
            orientation = pyro.sample("orientation", dist.Dirichlet(alpha))

            concentration = self.selectivity_p_concentration.expand(B, 1)
            rate = self.selectivity_p_rate.expand(B, 1)
            selectivity = orientation * pyro.sample(
                "selectivity", dist.Gamma(concentration, rate).to_event(1)
            )

        if "surprise" in self.ablations:
            surprise = regressors.new_zeros(torch.Size((P, B, 1)))
        else:
            concentration = self.surprise_p_concentration.expand(B, 1)
            rate = self.surprise_p_rate.expand(B, 1)
            surprise = pyro.sample("surprise",
                                   dist.Gamma(concentration, rate).to_event(1))

        baseline = baseline.unsqueeze(-2).expand(*baseline.shape[:2], 4, 1)

        coefficients = torch.cat((orientation * selectivity, -adaptation,
                                  surprise), dim=-1)[:,:,self.regressor_indices]
        regressors = regressors.expand(P, *regressors.shape)
        coefficients = coefficients.unsqueeze(-2).expand(*regressors.shape)
        predictions = torch.linalg.vecdot(coefficients, regressors)
        predictions = predictions.unsqueeze(dim=-1) + baseline
        pyro.sample("MUAe", dist.Normal(predictions, 0.1).to_event(2),
                    obs=muae.unsqueeze(dim=0) if muae is not None else None)

        return predictions

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
