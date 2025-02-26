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
        data = torch.cat((muae, regressors), dim=-1)

        alpha = self.oddball_alpha.expand(muae.shape[1], 3)
        oddball = pyro.sample("oddball?", dist.Dirichlet(alpha).to_event(1))

        alpha = self.orientation_alpha.expand(muae.shape[1], 2)
        orientation = pyro.sample("orientation",
                                  dist.Dirichlet(alpha).to_event(1))

        if "adaptation" in self.ablations:
            adaptation = data.new_zeros(torch.Size((orientation.shape[0],
                                                    *muae.shape, 1)))
        else:
            loc = self.adaptation_q_loc.expand(muae.shape[1], 1)
            log_scale = self.adaptation_q_log_scale.expand(muae.shape[1], 1)
            adaptation_dist = dist.LogNormal(loc, log_scale.exp())
            adaptation = pyro.sample("adaptation", adaptation_dist.to_event(2))

        alpha = self.block_alpha.expand(muae.shape[1], 3)
        block = pyro.sample("block", dist.Dirichlet(alpha).to_event(1))

        if "surprise" in self.ablations:
            surprise = data.new_zeros(torch.Size((block.shape[0],
                                                  *muae.shape[:2], 4)))
        else:
            loc = self.surprise_q_loc.expand(muae.shape[1], 4)
            log_scale = self.surprise_q_log_scale.expand(muae.shape[1], 4)
            surprise = pyro.sample("surprise",
                                   dist.LogNormal(loc,
                                                  log_scale.exp()).to_event(2))

        loc, log_scale = self.baseline_params(data).unbind(dim=-1)
        baseline = pyro.sample("baseline", dist.Normal(
            loc.unsqueeze(dim=-1), log_scale.exp().unsqueeze(dim=-1)
        ).to_event(2))

    def model(self, muae, regressors):
        # regressors[:, 0:2] = one-hot for oddball status
        # regressors[:, 3:4] = one-hot for orientation (0 -> 45, 1 -> 135)
        # regressors[:, 5] = stimulus repetition count up to current stimulus
        # regressors[:, 6:8] = one-hot for main, random control, seq control
        # regressors[:, 9:12] = surprisals
        concentration = muae.new_ones(torch.Size((*muae.shape[:2], 3)))
        oddball = pyro.sample("oddball?",
                              dist.Dirichlet(concentration).to_event(1))

        concentration = muae.new_ones(torch.Size((*muae.shape[:2], 2)))
        orientation = pyro.sample("orientation",
                                  dist.Dirichlet(concentration).to_event(1))

        if "adaptation" in self.ablations:
            adaptation = muae.new_zeros(torch.Size((oddball.shape[0],
                                                    *muae.shape[:2], 1)))
        else:
            loc = self.adaptation_p_loc.expand(*muae.shape[:2], 1)
            log_scale = self.adaptation_p_log_scale.expand(*muae.shape[:2], 1)
            adaptation = pyro.sample("adaptation", dist.LogNormal(
                loc, log_scale.exp()
            ).to_event(2))

        concentration = muae.new_ones(torch.Size((*muae.shape[:2], 3)))
        block = pyro.sample("block", dist.Dirichlet(concentration).to_event(1))

        if "surprise" in self.ablations:
            surprise = muae.new_zeros(torch.Size((oddball.shape[0],
                                                  *muae.shape[:2], 4)))
        else:
            loc = self.surprise_p_loc.expand(*muae.shape[:2], 4)
            scale = self.surprise_p_log_scale.expand(*muae.shape[:2], 4).exp()
            surprise = pyro.sample("surprise",
                                   dist.LogNormal(loc, scale).to_event(2))

        loc = muae.new_zeros(torch.Size((*muae.shape[:2], 1)))
        scale = muae.new_ones(torch.Size((*muae.shape[:2], 1)))
        baseline = pyro.sample("baseline", dist.Normal(loc, scale).to_event(2))

        coefficients = torch.cat((oddball, orientation, -adaptation, block,
                                  surprise), dim=-1)
        regressors = regressors.unsqueeze(dim=0).expand(coefficients.shape[0],
                                                        *regressors.shape)
        predictions = torch.linalg.vecdot(coefficients, regressors)
        predictions = predictions.unsqueeze(dim=-1) + baseline
        pyro.sample("MUAe", dist.Normal(predictions,
                                        self.log_scale.exp()).to_event(2),
                    obs=muae)

        return predictions
