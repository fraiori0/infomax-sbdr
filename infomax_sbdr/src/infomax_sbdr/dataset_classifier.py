import os
import numpy as onp
import jax.numpy as np
from torch.utils.data import Dataset
import pickle
from typing import Callable, Sequence
import torch
from torchvision import tv_tensors


class ClassifierDataset(Dataset):
    def __init__(
        self,
        x: onp.ndarray,
        categorical_labels: onp.ndarray,
        transform: Callable = None,
        device = None,
    ) -> None:

        self.transform = transform
        self.device = device

        # Perform one-hot encoding of the labels
        self.labels = onp.eye(categorical_labels.max()+1)[categorical_labels]

        # Store as torch tensors
        self.x = torch.from_numpy(x)
        self.labels = torch.from_numpy(self.labels)

        # Move to the specific device, if specified
        if device is not None:
            self.x = self.x.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = self.x[idx]
        label = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, label