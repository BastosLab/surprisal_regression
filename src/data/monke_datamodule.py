import glob
import hdf5storage as mat
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

class MuaTimeseries:
    def __init__(self, path):
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
    def __init__(self, data_dir):
        super().__init__()
        self._files = glob.glob(data_dir + "/*.mat")
        self._num_elements = 0
        for path in self._files:
            mua_info = mat.loadmat(path, squeeze_me=True)['datastruct']
            mua_info = dict(zip(mua_info.dtype.names, mua_info.item()))
            mua_info['areas']= [area.item() for area in
                                mua_info['areas'][:, 0].tolist()]
            mua_info['muae'] = dict(zip(mua_info['areas'],
                                        mua_info['muae'][0, :].tolist()))
            num_areas = sum([mua_info['muae'][area] is not None for area in
                             mua_info['areas']])
            num_trials = mua_info['stim_info'].shape[0]
            self._num_elements += num_areas * num_trials
            del mua_info

    def __iter__(self):
        for path in self._files:
            mua_info = mat.loadmat(path, squeeze_me=True)['datastruct']
            mua_info = dict(zip(mua_info.dtype.names, mua_info.item()))
            mua_info['areas']= [area.item() for area in
                                mua_info['areas'][:, 0].tolist()]
            mua_info['muae'] = dict(zip(mua_info['areas'],
                                        mua_info['muae'][0, :].tolist()))
            mua_info['session'] = mua_info['session'].item()
            mua_info['times_in_trial'] = dict(zip(
                mua_info['areas'], mua_info['times_in_trial'][0, :].tolist()
            ))
            for area in mua_info['areas']:
                if len(mua_info['muae'][area]):
                    for trial in range(mua_info['stim_info'].shape[0]):
                        muae = mua_info['muae'][area][:, :, trial]
                        times = mua_info['times_in_trial'][area]
                        stim_info = mua_info['stim_info'][trial, :, :]
                        yield (muae, times, stim_info, mua_info['stim_times'])

    def __len__(self):
        return self._num_elements

class MuaMatDataModule(LightningDataModule):
    def __init__(
        self, data_dir: str, area: str,
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
            dataset = MuaTimeseriesDataset(self.hparams.data_dir,
                                           self.hparams.area)
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
