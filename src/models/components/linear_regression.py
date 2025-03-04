import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
from torch import nn

from . import pyro as base

class TrialwiseLinearRegression(base.PyroModel):
    def __init__(self, ablations=[], hidden_dims=128, num_regressors=13,
                 num_stimuli=4):
        super().__init__()
        self._ablations = ablations
        self._num_regressors = num_regressors
        self._num_stimuli = num_stimuli

        self.log_scale = nn.Parameter(torch.tensor(0.))

        if "adaptation" not in self.ablations:
            self.adaptation_q_loc = pnn.PyroParam(torch.zeros(1))
            self.adaptation_q_log_scale = pnn.PyroParam(torch.zeros(1))

            self.adaptation_p_loc = pnn.PyroParam(torch.zeros(1))
            self.adaptation_p_log_scale = pnn.PyroParam(torch.zeros(1))
        self.baseline_params = nn.Sequential(
            nn.Linear(num_regressors + 1, hidden_dims), nn.SiLU(),
            nn.Linear(hidden_dims, 2)
        )

        self.block_alpha = pnn.PyroParam(torch.ones(3),
                                         constraint=dist.constraints.simplex)
        self.oddball_alpha = pnn.PyroParam(torch.ones(3),
                                           constraint=dist.constraints.simplex)
        self.orientation_alpha = pnn.PyroParam(torch.ones(2),
                                               constraint=dist.constraints.simplex)
        if "surprise" not in self.ablations:
            self.surprise_q_loc = pnn.PyroParam(torch.zeros(4))
            self.surprise_q_log_scale = pnn.PyroParam(torch.zeros(4))

            self.surprise_p_loc = pnn.PyroParam(torch.zeros(4))
            self.surprise_p_log_scale = pnn.PyroParam(torch.zeros(4))

    @property
    def ablations(self):
        return self._ablations

    def guide(self, muae, regressors):
        B = muae.shape[0]
        data = torch.cat((muae, regressors), dim=-1)

        alpha = self.oddball_alpha.expand(B, 3)
        oddball = pyro.sample("oddball?", dist.Dirichlet(alpha))
        alpha = self.orientation_alpha.expand(B, 2)
        orientation = pyro.sample("orientation", dist.Dirichlet(alpha))

        if "adaptation" in self.ablations:
            adaptation = data.new_zeros(torch.Size((orientation.shape[0],
                                                    *muae.shape, 1)))
        else:
            loc = self.adaptation_q_loc.expand(B, 1)
            log_scale = self.adaptation_q_log_scale.expand(B, 1)
            adaptation_dist = dist.LogNormal(loc, log_scale.exp()).to_event(1)
            adaptation = pyro.sample("adaptation", adaptation_dist)

        alpha = self.block_alpha.expand(B, 3)
        block = pyro.sample("block", dist.Dirichlet(alpha))

        if "surprise" in self.ablations:
            surprise = data.new_zeros(torch.Size((block.shape[0],
                                                  *muae.shape[:2], 4)))
        else:
            loc = self.surprise_q_loc.expand(B, 4)
            log_scale = self.surprise_q_log_scale.expand(B, 4)
            surprise = pyro.sample("surprise", dist.LogNormal(
                loc, log_scale.exp()
            ).to_event(1))

        loc, log_scale = self.baseline_params(data).mean(dim=1).unbind(dim=-1)
        baseline = pyro.sample("baseline", dist.Normal(
            loc.unsqueeze(dim=-1), log_scale.exp().unsqueeze(dim=-1)
        ).to_event(1))

    def model(self, muae, regressors):
        # regressors[:, 0:2] = one-hot for oddball status
        # regressors[:, 3:4] = one-hot for orientation (0 -> 45, 1 -> 135)
        # regressors[:, 5] = stimulus repetition count up to current stimulus
        # regressors[:, 6:8] = one-hot for main, random control, seq control
        # regressors[:, 9:12] = surprisals
        B = muae.shape[0]
        concentration = muae.new_ones(torch.Size((B, 3)))
        oddball = pyro.sample("oddball?", dist.Dirichlet(concentration))

        concentration = muae.new_ones(torch.Size((B, 2)))
        orientation = pyro.sample("orientation", dist.Dirichlet(concentration))

        if "adaptation" in self.ablations:
            adaptation = muae.new_zeros(torch.Size((oddball.shape[0],
                                                    *muae.shape[:1], 1)))
        else:
            loc = self.adaptation_p_loc.expand(B, 1)
            log_scale = self.adaptation_p_log_scale.expand(B, 1)
            adaptation = pyro.sample("adaptation", dist.LogNormal(
                loc, log_scale.exp()
            ).to_event(1))

        concentration = muae.new_ones(torch.Size((B, 3)))
        block = pyro.sample("block", dist.Dirichlet(concentration))

        if "surprise" in self.ablations:
            surprise = muae.new_zeros(torch.Size((oddball.shape[0],
                                                  *muae.shape[:1], 4)))
        else:
            loc = self.surprise_p_loc.expand(B, 4)
            log_scale = self.surprise_p_log_scale.expand(B, 4)
            surprise = pyro.sample("surprise", dist.LogNormal(
                loc, log_scale.exp()
            ).to_event(1))

        loc = muae.new_zeros(torch.Size((B, 1,)))
        scale = muae.new_ones(torch.Size((B, 1,)))
        baseline = pyro.sample("baseline", dist.Normal(loc, scale).to_event(1))
        baseline = baseline.unsqueeze(-2).expand(*baseline.shape[:2], 4, 1)

        coefficients = torch.cat((oddball, orientation, -adaptation, block,
                                  surprise), dim=-1)
        regressors = regressors.expand(coefficients.shape[0], *regressors.shape)
        coefficients = coefficients.unsqueeze(-2).expand(*regressors.shape)
        predictions = torch.linalg.vecdot(coefficients, regressors)
        predictions = predictions.unsqueeze(dim=-1) + baseline
        pyro.sample("MUAe", dist.Normal(predictions,
                                        self.log_scale.exp()).to_event(2),
                    obs=muae.unsqueeze(dim=0))

        return predictions
