from typing import Any, Dict, Tuple

from lightning import LightningModule
import pyro
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from .components import pyro as base
from .svi_module import SviLightningModule

class MonkeLightningModule(SviLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=['importance'])

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        muae, regressors = batch
        muae, regressors = muae.to(torch.float), regressors.to(torch.float)
        P = self.criterion.num_particles
        with pyro.plate("batch", muae.shape[0]):
            loss = self.elbo(muae, regressors)

        with torch.no_grad():
            with pyro.plate_stack("predictions", (P, muae.shape[0])):
                predictions, _, log_weights = self.forward(muae, regressors)
        return loss, predictions, log_weights
