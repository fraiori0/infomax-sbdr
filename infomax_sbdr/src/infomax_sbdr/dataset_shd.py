
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable
import gzip
import io

class SHDDataset(Dataset):
    """
    Returns spike trains as dense tensors of shape (n_steps, n_units).
    
    Args:
        path:        path to the .h5 file (e.g. 'shd_train.h5')
        n_steps:     number of time bins to discretize into
        n_units:     number of input channels (700 for SHD)
        t_start:     start of the time window (seconds)
        t_end:       end of the time window (seconds)
        unit_max:    if True, clip spike counts to 1 (binary spikes per bin)
    """
    def __init__(
        self,
        folder_path: str,
        kind: str = "train",
        num_steps: int = 100,
        n_units: int = 700,
        t_start: float = 0.0,
        t_end: float = 1.2,
        unit_max: bool = True,
    ):
        
        if kind not in ["train", "test"]:
            raise ValueError("kind must be 'train' or 'test'")
        
        super().__init__()
        
        self.folder_path = folder_path
        self.kind = kind
        self.num_steps = num_steps
        self.n_units = n_units
        self.t_start = t_start
        self.t_end   = t_end
        self.unit_max = unit_max

        filepath = os.path.join(
            folder_path,
            f"shd_{kind}.h5",
        )

        if not os.path.exists(filepath):
            raise ValueError(
                f"SHD dataset not found at {filepath}. "
        )

        with h5py.File(filepath, "r") as f:
            # Load everything into RAM (dataset is small enough, ~1 GB)
            self.labels = f["labels"][()]                      # (N,)
            times  = f["spikes/times"][()]                     # object array of arrays
            units  = f["spikes/units"][()]                     # object array of arrays

        self.data = self._encode(times, units)                 # (N, num_steps, n_units)

        # print timestep ranges
        t_min, t_max = np.array(self.stats["t_min"]), np.array(self.stats["t_max"])
        print(f"Time range: [{t_min.mean():.3f}({t_min.std():.3f}), {t_max.mean():.3f} ({t_max.std():.3f})]")
        print(f"\tT_max range: {t_max.min():.3f} - {t_max.max():.3f}")


    def _encode(self, all_times, all_units):
        N = len(all_times)
        out = np.zeros((N, self.num_steps, self.n_units), dtype=np.float32)

        dt = (self.t_end - self.t_start) / self.num_steps       # seconds per bin

        self.stats = {
            "t_min": [],
            "t_max": [],
            "n_spikes": [],
        }

        for i, (times, units) in enumerate(zip(all_times, all_units)):
            # Convert spike times to bin indices
            t_shifted = times - self.t_start
            bin_idx = (t_shifted / dt).astype(int)

            # Keep only spikes within [t_start, t_end)
            mask = (bin_idx >= 0) & (bin_idx < self.num_steps)
            bin_idx = bin_idx[mask]
            unit_idx = units[mask].astype(int)

            # Keep only valid unit indices
            valid = (unit_idx >= 0) & (unit_idx < self.n_units)
            bin_idx  = bin_idx[valid]
            unit_idx = unit_idx[valid]

            np.add.at(out[i], (bin_idx, unit_idx), 1.0)

            if self.unit_max:
                np.clip(out[i], 0, 1, out=out[i])             # binarize

            self.stats["t_min"].append(t_shifted.min())
            self.stats["t_max"].append(t_shifted.max())

        return out

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spikes = self.data[idx]              # (n_steps, n_units)
        label  = self.labels[idx]

        # # add a small amount of salt-pepper noise 
        # if self.kind == "train":
        #     # pos_noise = np.random.rand(*spikes.shape) < 0.01
        #     neg_mask = np.random.rand(*spikes.shape) < 0.9
        #     spikes = spikes * neg_mask
            

        return spikes, label