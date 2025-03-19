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

        if "repetition" not in self.ablations:
            self.repetition_q_concentration = pnn.PyroParam(
                torch.ones(1), constraint=dist.constraints.positive
            )
            self.repetition_q_rate = pnn.PyroParam(
                torch.ones(1), constraint=dist.constraints.positive
            )
            self.register_buffer("adaptation_concentration", torch.ones(1))
            self.register_buffer("adaptation_rate", torch.ones(1))

        self.angle_q_alpha = pnn.PyroParam(torch.ones(2),
                                           constraint=dist.constraints.simplex)
        self.register_buffer("angle_p_alpha", torch.ones(2))

        self.baseline_q = nn.Sequential(
            nn.Linear(num_stimuli * (num_regressors + 1), hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, 2)
        )
        self.register_buffer("baseline_p_loc", torch.zeros(1))
        self.register_buffer("baseline_p_log_scale", torch.zeros(1))

        if "selectivity" not in self.ablations:
            self.selectivity_q_concentration = pnn.PyroParam(
                torch.ones(1), constraint=dist.constraints.positive
            )
            self.selectivity_q_rate = pnn.PyroParam(
                torch.ones(1), constraint=dist.constraints.positive
            )
            self.register_buffer("selectivity_p_concentration", torch.ones(1))
            self.register_buffer("selectivity_p_rate", torch.ones(1))

        if "surprise" not in self.ablations:
            self.surprise_q_concentration = pnn.PyroParam(
                torch.ones(4), constraint=dist.constraints.simplex
            )
            self.surprise_q_rate = pnn.PyroParam(
                torch.ones(4), constraint=dist.constraints.simplex
            )
            self.register_buffer("surprise_p_concentration", torch.ones(4))
            self.register_buffer("surprise_p_rate", torch.ones(4))

    @property
    def ablations(self):
        return self._ablations

    def guide(self, muae, regressors):
        B = muae.shape[0]
        data = torch.cat((muae, regressors), dim=-1)
        loc, log_scale = self.baseline_q(data.flatten(-2, -1)).unbind(dim=-1)
        baseline = pyro.sample("baseline", dist.Normal(
            loc.unsqueeze(dim=-1), log_scale.exp().unsqueeze(dim=-1)
        ).to_event(1))
        P = baseline.shape[0]

        if "selectivity" not in self.ablations:
            alpha = self.angle_alpha.expand(B, 2)
            orientation = pyro.sample("orientation", dist.Dirichlet(alpha))

            concentration = self.selectivity_q_concentration.expand(B, 1)
            rate = self.selectivity_q_rate.expand(B, 1)
            selectivity = orientation * pyro.sample(
                "selectivity", dist.Gamma(concentration, rate).to_event(1)
            )

        if "repetition" not in self.ablations:
            concentration = self.repetition_q_concentration.expand(B, 1)
            rate = self.repetition_q_rate.expand(B, 1)
            repetition_dist = dist.Gamma(concentration, rate).to_event(1)
            repetition = pyro.sample("repetition", repetition_dist)

        if "surprise" not in self.ablations:
            concentration = self.surprise_q_concentration.expand(B, 4)
            rate = self.surprise_q_rate.expand(B, 4)
            surprise = pyro.sample("surprise",
                                   dist.Gamma(concentration, rate).to_event(1))

    def model(self, muae, regressors):
        # regressors[:, 0:1] = one-hot for orientation (0 -> 45, 1 -> 135)
        # regressors[:, 2] = stimulus repetition count up to current stimulus
        # regressors[:, 3:6] = surprisals
        B = regressors.shape[0]
        loc = self.baseline_p_loc.expand(B, 1)
        log_scale = self.baseline_p_log_scale.expand(B, 1)
        baseline = pyro.sample("baseline",
                               dist.Normal(loc, log_scale.exp()).to_event(1))
        P = baseline.shape[0]

        if "repetition" in self.ablations:
            repetition = regressors.new_zeros(torch.Size((P, B, 1)))
        else:
            concentration = self.repetition_p_concentration.expand(B, 1)
            rate = self.repetition_p_rate.expand(B, 1)
            repetition = pyro.sample(
                "repetition", dist.Gamma(concentration, rate).to_event(1)
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
            surprise = regressors.new_zeros(torch.Size((P, B, 4)))
        else:
            concentration = self.surprise_p_concentration.expand(B, 4)
            rate = self.surprise_p_rate.expand(B, 4)
            surprise = pyro.sample("surprise",
                                   dist.Gamma(concentration, rate).to_event(1))

        baseline = baseline.unsqueeze(-2).expand(*baseline.shape[:2], 4, 1)

        coefficients = torch.cat((orientation * selectivity, -repetition,
                                  surprise), dim=-1)
        regressors = regressors.expand(coefficients.shape[0], *regressors.shape)
        coefficients = coefficients.unsqueeze(-2).expand(*regressors.shape)
        predictions = torch.linalg.vecdot(coefficients, regressors)
        predictions = predictions.unsqueeze(dim=-1) + baseline
        pyro.sample("MUAe", dist.Normal(predictions, 0.1).to_event(2),
                    obs=muae.unsqueeze(dim=0) if muae is not None else None)

        return predictions
