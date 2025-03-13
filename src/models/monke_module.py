from typing import Any, Dict, Tuple

from lightning import LightningModule
import pyro
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from .components import pyro as base
from .svi_module import SviLightningModule

class MonkeLightningModule(SviLightningModule):
    pass
