import os
import numpy as onp
import jax.numpy as np
from torch.utils.data import Dataset
import pickle
from typing import Callable, Sequence
import torch
from torchvision import tv_tensors

class CustomImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset (datasets.Dataset): The Hugging Face dataset object (e.g., hf_train_dataset).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Get an item from the Hugging Face dataset
        item = self.hf_dataset[idx]

        # The 'image' column usually contains a PIL.Image.Image object
        image = item['image']
        # The 'label' column contains the integer class ID
        label = item['label']

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label