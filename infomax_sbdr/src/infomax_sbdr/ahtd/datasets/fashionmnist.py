"""FashionMNIST dataset with Poisson spike encoding."""

import numpy as np
from pathlib import Path
import gzip

import torch
from torch.utils.data import Dataset


def load_fashionmnist(data_dir, kind='train'):
    """Load FashionMNIST data from ubyte files."""
    data_dir = Path(data_dir)
    
    if kind == 'train':
        labels_path = data_dir / 'train-labels-idx1-ubyte.gz'
        images_path = data_dir / 'train-images-idx3-ubyte.gz'
    else:
        labels_path = data_dir / 't10k-labels-idx1-ubyte.gz'
        images_path = data_dir / 't10k-images-idx3-ubyte.gz'
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    return images, labels


class FashionMNISTPoissonDataset(Dataset):
    """FashionMNIST with Poisson spike encoding."""
    
    def __init__(
        self,
        folder_path,
        kind='train',
        n_timesteps=30,
        dt=1.0,
        max_rate=100.0,
        flatten=True,
        transform=None,
        normalize=True,
    ):
        self.folder_path = folder_path
        self.kind = kind
        self.n_timesteps = n_timesteps
        self.dt = dt
        self.max_rate = max_rate
        self.flatten = flatten
        self.transform = transform
        self.normalize = normalize
        
        self.images, self.labels = load_fashionmnist(folder_path, kind)
        
        if normalize:
            self.images = self.images.astype(np.float32) / 255.0
        else:
            self.images = self.images.astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Poisson spike encoding
        rates = image * self.max_rate
        spike_prob = rates * (self.dt / 1000.0)
        spikes = np.random.rand(self.n_timesteps, *image.shape) < spike_prob
        spikes = spikes.astype(np.float32)
        
        if self.flatten and spikes.ndim > 2:
            spikes = spikes.reshape(self.n_timesteps, -1)
        
        return spikes, label


def create_fashionmnist_dataloaders(
    data_dir,
    batch_size=32,
    val_split=0.2,
    n_timesteps=30,
    dt=1.0,
    max_rate=100.0,
    shuffle_train=True,
    seed=42,
):
    """Create FashionMNIST train/val dataloaders."""
    import sys
    from pathlib import Path
    
    # Try to import NumpyLoader
    try:
        from antihebbian_td.utils.torch_dataloader import NumpyLoader
    except ImportError:
        # Fallback if package not installed
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from antihebbian_td.utils.torch_dataloader import NumpyLoader
    
    dataset_train_full = FashionMNISTPoissonDataset(
        folder_path=data_dir,
        kind='train',
        n_timesteps=n_timesteps,
        dt=dt,
        max_rate=max_rate,
        flatten=True,
        normalize=True,
    )
    
    n_val = int(len(dataset_train_full) * val_split)
    n_train = len(dataset_train_full) - n_val
    
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    
    dataset_test = FashionMNISTPoissonDataset(
        folder_path=data_dir,
        kind='t10k',
        n_timesteps=n_timesteps,
        dt=dt,
        max_rate=max_rate,
        flatten=True,
        normalize=True,
    )
    
    train_loader = NumpyLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
    val_loader = NumpyLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = NumpyLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader
