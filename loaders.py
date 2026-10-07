#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

def _h5_to_complex(arr):
    """
    Convert h5py-read MATLAB complex array to numpy complex array.
    Supports compound dtype with real/imag fields.
    """
    arr = np.asarray(arr)

    if arr.dtype.names is not None:
        names = arr.dtype.names

        if "real" in names and "imag" in names:
            arr = arr["real"] + 1j * arr["imag"]
        elif "r" in names and "i" in names:
            arr = arr["r"] + 1j * arr["i"]
        else:
            raise ValueError(f"Unknown complex dtype fields: {names}")

    return np.asarray(arr, dtype=np.complex64)

def load_first_subcarrier_from_revised_mat(filename):
    """
    Fast loader for MATLAB -v7.3 .mat files.

    It only reads output_h, and only reads the first subcarrier.
    It avoids hdf5storage.loadmat, so it will not parse gt_path_meta.

    Expected MATLAB output_h:
        [num_channels, selected_sc, 32, 32]

    h5py usually sees MATLAB v7.3 array as:
        [32, 32, selected_sc, num_channels]

    Returned shape:
        [num_channels, 32, 32]
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"MAT file not found: {filename}")

    with h5py.File(filename, "r") as f:
        if "output_h" not in f:
            raise KeyError(f"'output_h' not found in {filename}. Available keys: {list(f.keys())}")

        ds = f["output_h"]
        shape = ds.shape

        if len(shape) != 4:
            raise ValueError(f"output_h should be 4-D, but got shape {shape}")

        # Case 1: h5py reading MATLAB v7.3 layout, usually [32, 32, K, N]
        if shape[0] == 32 and shape[1] == 32:
            raw = ds[:, :, 0, :]
            raw = _h5_to_complex(raw)

            # Convert to [N, 32, 32]
            channels = np.transpose(raw, (2, 1, 0))

        # Case 2: already [N, K, 32, 32]
        else:
            raw = ds[:, 0, :, :]
            channels = _h5_to_complex(raw)

    return np.ascontiguousarray(channels, dtype=np.complex64)

class Channels(Dataset):
    """MIMO Channels"""

    def __init__(self, seed, config, norm=None, num_rx=1024, mask_ratios=[0.2]):
        target_spacings = config.data.spacing_list
        target_channel = config.data.channel

        self.channels = []
        self.spacings = np.copy(target_spacings)
        self.filenames = []
        self.mask_ratios = mask_ratios
        self.num_rx = num_rx

        for spacing in target_spacings:
            filename = './data/%s_Nt1_Nr%d_QuaRIGA_UPA%.2f_seed%d.mat' % (
                    target_channel, self.num_rx, spacing, seed
            )
            self.filenames.append(filename)

            t0 = time.time()
            channels_first_sc = load_first_subcarrier_from_revised_mat(filename)
            print(
                f"[Channels] Loaded output_h first subcarrier from {filename} "
                f"with shape {channels_first_sc.shape} in {time.time() - t0:.2f} s",
                flush=True
            )

            self.channels.append(channels_first_sc)

        self.channels = np.asarray(self.channels)
        self.channels = np.reshape(
            self.channels,
            (-1, self.channels.shape[-2], self.channels.shape[-1])
        )

        # Normalize
        if type(norm) == list:
            self.mean = norm[0]
            self.std = norm[1]
        elif norm == 'entrywise':
            self.mean = np.mean(self.channels, axis=0)
            self.std = np.std(self.channels, axis=0)
        elif norm == 'global':
            self.mean = 0.
            self.std = np.std(self.channels)
        else:
            self.mean = 0.
            self.std = 1.

        self.std = np.where(np.abs(self.std) < 1e-12, 1.0, self.std)

        print(
            f"[Channels] Final channels shape: {self.channels.shape}, "
            f"mean type: {type(self.mean)}, std type: {type(self.std)}",
            flush=True
        )

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        H_cplx = self.channels[idx]
        H_cplx_norm = (H_cplx - self.mean) / self.std

        mask_ratio = np.random.choice(self.mask_ratios)

        # Broadcast the 2D observation mask to real and imaginary components.
        mask = np.random.rand(*H_cplx_norm.shape) > mask_ratio

        H_real_norm_full = np.stack(
            (np.real(H_cplx_norm), np.imag(H_cplx_norm)),
            axis=0
        )

        H_real_norm_partial = H_real_norm_full * mask

        sample = {
            'H_full': H_real_norm_full.astype(np.float32),
            'H_partial': H_real_norm_partial.astype(np.float32),
            'mask_ratio': mask_ratio,
            'mask': mask.astype(np.float32),
            'idx': int(idx)
        }

        return sample
