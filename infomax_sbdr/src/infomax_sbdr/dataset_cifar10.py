import os
import numpy as onp
import jax.numpy as np
from torch.utils.data import Dataset
import pickle
from typing import Callable, Sequence
import torch
from torchvision import tv_tensors


class Cifar10Dataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        kind: str = "train",
        transform: Callable = None,
        device = None,
    ) -> None:

        super().__init__()

        if kind not in ["train", "test"]:
            raise ValueError("kind must be either 'train' or 'test'")

        self.folder_path = folder_path
        self.transform = transform

        # NOTE: the names written here correspond to those of the folder downloadable from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        self.file_names = {
            "train": (
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ),
            "test": ("test_batch"),
        }

        # Load all data of the given kind
        data = {
            "data": [],
            "labels": [],
        }
        for file_name in self.file_names[kind]:
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, "rb") as f:
                d = pickle.load(f, encoding="bytes")
                data["data"].append(d[b"data"])
                data["labels"].append(d[b"labels"])

        self.images = onp.concatenate(data["data"], axis=0, dtype=onp.uint8)
        self.labels = onp.concatenate(data["labels"], axis=0, dtype=onp.int32)

        # Perform one-hot encoding of the labels
        self.labels = onp.eye(10)[self.labels]

        # Reshape images to be (C, H, W)
        # the original from file are already flattened and are in row-major order
        self.images = self.images.reshape(self.images.shape[0], 3, 32, 32)

        # convert to torchvision image
        self.images = tv_tensors.Image(self.images)
        # and to torch array for the labels
        self.labels = torch.from_numpy(self.labels)

        # convert to float and divide by 255
        self.images = self.images.float() / 255

        # # print the mean and std for each channel
        # print(f"CIFAR10 Dataset Loaded ({kind})")
        # print(f"\tChannel Mean: {self.images.mean(dim=(0, 1, 2))}")
        # print(f"\tChannel Std: {self.images.std(dim=(0, 1, 2))}")

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

        # print the device on which the image is

        return img, label


class Cifar10DatasetContrastive(Cifar10Dataset):
    def __init__(
        self,
        folder_path: str,
        kind: str = "train",
        transform: Callable = None,
        device = None,
    ) -> None:
        super().__init__(folder_path, kind, transform, device)

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

        return (img_1, img_2), label
