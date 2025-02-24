import pyro
import pyro.distributions as dist
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
            self.adaptation_params = nn.Sequential(
                nn.Linear(num_regressors + 1, hidden_dims), nn.SiLU(),
                nn.Linear(hidden_dims, 2)
            )
        self.baseline_params = nn.Sequential(
            nn.Linear(num_regressors + 1, hidden_dims), nn.SiLU(),
            nn.Linear(hidden_dims, 2)
        )
        self.block_alpha = nn.Sequential(
            nn.Linear(num_regressors + 1, hidden_dims), nn.SiLU(),
            nn.Linear(hidden_dims, 3), nn.Softmax(dim=-1)
        )
        self.oddball_alpha = nn.Sequential(
            nn.Linear(num_regressors + 1, hidden_dims), nn.SiLU(),
            nn.Linear(hidden_dims, 3), nn.Softmax(dim=-1)
        )
        self.orientation_alpha = nn.Sequential(
            nn.Linear(num_regressors + 1, hidden_dims), nn.SiLU(),
            nn.Linear(hidden_dims, 2), nn.Softmax(dim=-1)
        )
        if "surprise" not in self.ablations:
            self.surprise_params = nn.Sequential(
                nn.Linear(num_regressors + 1, hidden_dims), nn.SiLU(),
                nn.Linear(hidden_dims, 4 * 2)
            )

    @property
    def ablations(self):
        return self._ablations

    def guide(self, muae, regressors):
        data = torch.cat((muae, regressors), dim=-1)
        concentration = self.oddball_alpha(data)
        oddball = pyro.sample("oddball?",
                              dist.Dirichlet(concentration).to_event(1))

        concentration = self.orientation_alpha(data)
        orientation = pyro.sample("orientation",
                                  dist.Dirichlet(concentration).to_event(1))

        if "adaptation" in self.ablations:
            adaptation = data.new_zeros(torch.Size((orientation.shape[0],
                                                    *muae.shape, 1)))
        else:
            loc, log_scale = self.adaptation_params(data).unbind(dim=-1)
            adaptation = pyro.sample("adaptation", dist.LogNormal(
                loc.unsqueeze(dim=-1), log_scale.exp().unsqueeze(dim=-1)
            ).to_event(2))

        concentration = self.block_alpha(data)
        block = pyro.sample("block", dist.Dirichlet(concentration).to_event(1))

        if "surprise" in self.ablations:
            surprise = data.new_zeros(torch.Size((block.shape[0],
                                                  *muae.shape[:2], 4)))
        else:
            loc, log_scale = self.surprise_params(data).view(
                data.shape[0], 4, 4, 2
            ).unbind(dim=-1)
            surprise = pyro.sample("surprise", dist.LogNormal(
                loc, log_scale.exp()
            ).to_event(2))

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

        loc = muae.new_zeros(torch.Size((*muae.shape[:2], 1)))
        if "adaptation" in self.ablations:
            adaptation = loc.unsqueeze(0).expand(oddball.shape[0], *loc.shape)
        else:
            scale = muae.new_ones(torch.Size((*muae.shape[:2], 1)))
            adaptation = pyro.sample("adaptation",
                                     dist.LogNormal(loc, scale).to_event(2))

        concentration = muae.new_ones(torch.Size((*muae.shape[:2], 3)))
        block = pyro.sample("block", dist.Dirichlet(concentration).to_event(1))

        loc = muae.new_zeros(torch.Size((*muae.shape[:2], 4)))
        if "surprise" in self.ablations:
            surprise = loc.unsqueeze(0).expand(oddball.shape[0], *loc.shape)
        else:
            scale = muae.new_ones(torch.Size((*muae.shape[:2], 4)))
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
