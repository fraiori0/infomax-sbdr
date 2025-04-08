import os
import jax.numpy as np
from torch.utils.data import Dataset
import gzip
from typing import Callable, Sequence


class FashionMNISTDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        kind: str = "train",  # 'train' or 'test'
        transform: Callable = None,
        flatten: bool = False,
    ) -> None:

        super().__init__()

        if kind not in ["train", "test"]:
            raise ValueError("kind must be 'train' or 'test'")

        self.folder_path = folder_path
        self.transform = transform

        # adjust the 'kind' string to match the file names for the FashionMNIST
        if kind == "test":
            kind = "t10k"

        # Load images and labels
        labels_path = os.path.join(self.folder_path, f"{kind}-labels-idx1-ubyte.gz")
        images_path = os.path.join(self.folder_path, f"{kind}-images-idx3-ubyte.gz")
        with gzip.open(labels_path, "rb") as lbpath:
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, "rb") as imgpath:
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            if flatten:
                self.images = self.images.reshape(len(self.labels), 784)
            else:
                # (N, C, H, W) format
                self.images = self.images.reshape(len(self.labels), 1, 28, 28)

        # Perform one-hot encoding of the labels
        self.labels = np.eye(10)[self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
