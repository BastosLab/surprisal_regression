import glob
import hdf5storage as mat
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

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
