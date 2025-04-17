import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable
from functools import partial


class DenseFLOSigmoid(nn.Module):
    """
    NN with final output given by n_active_out_features different softmax over different subset of the n_out_features
    each of dimension n_out_features//n_active_out_features
    """

    n_hid_features: Sequence[int]
    n_out_features: int
    activation_fn: Callable = nn.elu

    def setup(self):

        assert self.n_out_features % self.n_active_out_features == 0
        self.n_group_out_features = self.n_out_features // self.n_active_out_features

        layers = []
        for f in self.n_hid_features:
            layers.append(nn.Dense(features=f))
            layers.append(self.activation_fn)
        layers.append(nn.Dense(features=self.n_out_features))
        self.layers = nn.Sequential(layers)

    def __call__(self, x):
        # input x of shape (*batch_dims, features)
        # i.e., already flattened on the feature dimensions,
        # if there were multiple in the original data (like image C, H, W)

        # apply the feedforward weights
        x = self.layers(x)
        # perform softmax on reshaped output
        x = x.reshape(
            x.shape[:-1] + (self.n_active_out_features, self.n_group_out_features)
        )
        x = nn.softmax(x, axis=-1)
        # reshape back
        x = x.reshape(x.shape[:-2] + (self.n_out_features,))
        return x
