import os
import numpy as onp
import jax.numpy as np
from torch.utils.data import Dataset
import gzip
from typing import Callable, Sequence
import torch
from torchvision import tv_tensors


class FashionMNISTDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        kind: str = "train",  # 'train' or 'test'
        transform: Callable = None,
        flatten: bool = False,
        device = None,
        sequential = False
    ) -> None:
        
        if sequential and flatten:
            raise ValueError("\'sequential\' and \'flatten\' are mutually exclusive, they cannot both be True")

        super().__init__()

        if kind not in ["train", "test"]:
            raise ValueError("kind must be 'train' or 'test'")

        self.folder_path = folder_path
        self.transform = transform
        self.flatten = flatten
        self.device = device
        self.sequential = sequential

        # adjust the 'kind' string to match the file names for the FashionMNIST
        if kind == "test":
            kind = "t10k"

        # Load images and labels
        labels_path = os.path.join(self.folder_path, f"{kind}-labels-idx1-ubyte.gz")
        images_path = os.path.join(self.folder_path, f"{kind}-images-idx3-ubyte.gz")
        with gzip.open(labels_path, "rb") as lbpath:
            self.labels = onp.frombuffer(lbpath.read(), dtype=onp.uint8, offset=8)

        with gzip.open(images_path, "rb") as imgpath:
            self.images = onp.frombuffer(imgpath.read(), dtype=onp.uint8, offset=16)
            # (N, C, H, W) format
            self.images = self.images.reshape(len(self.labels), 1, 28, 28)
        
        # Perform one-hot encoding of the labels
        self.labels = onp.eye(10)[self.labels]

        # convert to torchvision image
        self.images = tv_tensors.Image(self.images)
        # and to torch array for the labels
        self.labels = torch.from_numpy(self.labels)

        # convert to float and divide by 255
        self.images = self.images.float() / 255

         # # print the mean and std for each channel
        print(f"FashionMNIST Dataset Loaded ({kind})")
        print(f"\tChannel Mean: {self.images.mean(dim=(0, 2, 3))}")
        print(f"\tChannel Std: {self.images.std(dim=(0, 2, 3))}")

        # Move to the specific device, if specified
        if device is not None:
            self.images = self.images.to(device)
            self.labels = self.labels.to(device)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.flatten:
            img = img.view(-1)
        
        if self.sequential:
            # return a sample of shape (28,28),
            # where rows are taken to be different time-stpes, and the columns are the actual input
            # i.e., the so-called Sequential FashionMNIST
            img = img.view(28, 28)

        return img, label


class FashionMNISTDatasetContrastive(FashionMNISTDataset):
    def __init__(
        self,
        folder_path: str,
        kind: str = "train",  # 'train' or 'test'
        transform: Callable = None,
        flatten: bool = False,
        device = None,
        sequential = False,
    ) -> None:
        super().__init__(
            folder_path=folder_path,
            kind=kind,
            transform=transform,
            flatten=flatten,
            device=device,
            sequential=sequential
        )

    # we change only the __getitem__ method
    # to apply data augmentation twice to the same image (to get two different versions)
    def __getitem__(self, idx):

        img = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            img_1 = self.transform(img)
            img_2 = self.transform(img)
        else:
            img_1 = img
            img_2 = img

        if self.flatten:
            img_1 = img_1.view(-1)
            img_2 = img_2.view(-1)

        return (img_1, img_2), label