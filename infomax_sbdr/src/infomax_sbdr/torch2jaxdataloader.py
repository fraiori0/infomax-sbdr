import numpy as onp
from jax.tree_util import tree_map
import jax.numpy as np
from torch.utils import data


"""
Utility functions to use a PyTorch dataloader with JAX

https://docs.jax.dev/en/latest/notebooks/Neural_Network_and_Data_Loading.html#data-loading-with-pytorch
"""


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
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
        super(self.__class__, self).__init__(
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


class FlattenAndCast(object):
    def __call__(self, pic):
        return onp.ravel(onp.array(pic, dtype=np.float32))
