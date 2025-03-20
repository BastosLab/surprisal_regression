import glob
import hdf5storage as mat
import numpy as np
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class MuaTimeseries:
    def __init__(self, path):
        log.info("Loading MUAe timeseries from %s" % path)
        raw = mat.loadmat(path, squeeze_me=True)['datastruct']
        raw = dict(zip(raw.dtype.names, raw.item()))
        self._possible_areas = [area.item() for area in
                                raw['areas'][:, 0].tolist()]
        self._muae = {k: v for k, v in zip(self._possible_areas,
                                           raw['muae'][0, :].tolist())
                      if len(v)}
        self._session = raw['session'].item()
        self._stim_info = raw['stim_info']
        self._stim_times = raw['stim_times']
        self._timestamps = {
            k: v for (k, v) in zip(self._possible_areas,
                                   raw['times_in_trial'][0, :].tolist())
            if len(v)
        }

    @property
    def areas(self):
        return self._muae.keys()

    def __len__(self):
        return self.stim_info.shape[0]

    @property
    def muae(self):
        return self._muae

    @property
    def session(self):
        return self._session

    @property
    def stim_info(self):
        return self._stim_info

    @property
    def stim_times(self):
        return self._stim_times

    @property
    def timestamps(self):
        return self._timestamps

class MuaTimeseriesDataset(IterableDataset):
    def __init__(self, session_path, area):
        super().__init__()
        self._area = area
        path, ext = os.path.splitext(session_path)
        assert ext == '.mat'
        assert os.path.isfile(session_path)
        self._file = session_path
        self._session = MuaTimeseries(path)
        assert self.area in self._session.muae.keys()
        assert np.isfinite(self._session.muae[self.area]).all()

    @property
    def area(self):
        return self._area

    def __getitem__(self, idx):
        muae = self._session.muae[self.area][:, :, idx]
        timestamps = self._session.timestamps[self.area]
        stim_info = self._session.stim_info[idx, :, :]
        stim_times = self._session.stim_times[idx, :, :]
        return (muae, timestamps, stim_info, stim_times)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self._session.muae[self.area].shape[-1]

class MuaPresentationDataset(IterableDataset):
    def __init__(self, session_path, area):
        super().__init__()
        self._timeseries = MuaTimeseriesDataset(session_path, area)

    def __getitem__(self, idx):
        muae, timestamps, stim_info, stim_times = self._timeseries[idx]
        stim_muae = []
        for s in range(1, 5):
            start, end = stim_times[s, 0], stim_times[s, 1] + 0.075
            start = np.nanargmin(np.abs(timestamps - start))
            end = np.nanargmin(np.abs(timestamps - end))
            stim_avg = muae[:, start:end+1].mean(axis=-1).mean(axis=0,
                                                               keepdims=True)
            stim_muae.append(stim_avg)
        # stim_info[:, 0] \in {0, 1, 2} for non-oddball, local, global
        # stim_info[:, 1] = grating orientation in degrees
        # stim_info[:, 2] \in {1, 2, 3} for main, random control, seq control
        # stim_info[:, 3] = conditional surprisal of this stimulus
        # stim_info[:, 4] = marginal surprisal of this stimulus
        # stim_info[:, 5] = cumulative conditional surprisal of this stimulus
        # stim_info[:, 6] = cumulative marginal surprisal of this stimulus
        angles = (stim_info[:, 1] == 135.).astype(int)
        angles = F.one_hot(torch.tensor(angles, dtype=torch.long), 2).numpy()
        adaptation = np.zeros(4)
        for p in range(1, 4):
            repeats = 0
            last = p - 1
            while last >= 0 and (angles[last, :] == angles[p, :]).all():
                adaptation[p] += 1
                last = last - 1
        adaptation = adaptation[:, np.newaxis]

        surprisals = stim_info[:, 3:]

        muae = np.stack(stim_muae, axis=0).astype(float)
        regressors = np.concatenate((angles, adaptation, surprisals), axis=-1)
        return muae, regressors

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self._timeseries)

class SyntheticMuaDataset(IterableDataset):
    def __init__(self, mat_path):
        super().__init__()
        self._regressors, self._samples = torch.load(mat_path)

    def __getitem__(self, idx):
        if idx < len(self._samples["lo"]):
            kind = "lo"
            r = 0
        elif idx < 2 * len(self._samples["lo"]):
            idx -= len(self._samples["lo"])
            kind = "go"
            r = 1
        elif idx < 3 * len(self._samples["lo"]):
            idx -= 2 * len(self._samples["lo"])
            kind = "rndctrl"
            r = 2
        elif idx < 4 * len(self._samples["lo"]):
            idx -= 3 * len(self._samples["lo"])
            kind = "seqctrl"
            r = 3

        return self._samples[kind][idx, :, :], self._regressors[r, :, :]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return sum(len(self._samples[type]) for type in self._samples.keys())

class MuaMatDataModule(LightningDataModule):
    def __init__(
        self, session_path: str, area: str,
        train_val_split: Tuple[float, float, float] = (0.8, 0.2),
        batch_size: int=64, num_workers: int = 0, pin_memory: bool = False
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.transforms = transforms.ToTensor()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = MuaPresentationDataset(self.hparams.session_path,
                                             self.hparams.area)
            self.data_train, self.data_val = random_split(
                dataset=dataset, lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_test = dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

class SyntheticMuaDataModule(LightningDataModule):
    def __init__(
        self, mat_path: str,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int=64, num_workers: int = 0, pin_memory: bool = False
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.transforms = transforms.ToTensor()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = SyntheticMuaDataset(self.hparams.mat_path)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset, lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
