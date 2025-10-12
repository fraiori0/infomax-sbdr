import numpy as onp
from torch.utils.data import Dataset
import torch
from typing import Callable


class XORDataset(Dataset):
    def __init__(
        self,
        device = None,
        seed = 0,
    ) -> None:
        
        super().__init__()

        self.device = device
        
        # for reproducibility
        self.rng = onp.random.default_rng(seed)

        # Generate the data
        x = self.rng.binomial(n=1, p=0.5, size=(5000, 2))
        # Generate the XOR
        y = onp.logical_xor(x[:, 0], x[:, 1])

        # Rescale inputs to be in [-0.5, 0.5]
        x = x - 0.5
        
        # convert to torch tensors
        self.x = torch.from_numpy(x.astype(onp.float32))
        self.y = torch.from_numpy(y.astype(onp.float32))

        # Move to the specific device, if specified
        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        x = self.x[idx]
        y = self.y[idx]

        return x, y
