"""Utility for using PyTorch dataloaders with JAX."""

import numpy as np
from jax.tree_util import tree_map
import jax.numpy as jnp
from torch.utils import data


def numpy_collate(batch):
    """Convert PyTorch batch to JAX arrays."""
    return tree_map(jnp.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    """PyTorch DataLoader that yields JAX arrays."""
    
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=False,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
        )


class FlattenAndCast:
    """Transform to flatten and cast images."""
    
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float32))
