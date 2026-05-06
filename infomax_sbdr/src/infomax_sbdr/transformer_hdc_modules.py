import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable, Tuple
from functools import partial
import infomax_sbdr.binary_comparisons as bc
import infomax_sbdr.utils as ut


""" Transformer-like Attention Modules """

class CyclicOuterLayer(nn.Module):
    """
    Module with a layers to generate keys and values.
    The queries and keys are generated using an outer product between the output of two separate layers:
        1. the first layer is a standard feedforward layer, which should extract features from the input.
        2. the second layer performs a cyclic permutation of its output, which should learn relative-timing information
    """
    features: int
    out_features: int

    def setup(self):
        # keys
        self.k_in = nn.Dense(self.features)
        self.k_pos = nn.Dense(self.features)
        # queries
        self.q_in = nn.Dense(self.features)
        self.q_pos = nn.Dense(self.features)
        # values
        self.v = nn.Dense(self.out_features)

    def __call__(self, x):
        # assume input x is of shape (*batch_dims, time, features)
        batch_shape = x.shape[:-2]
        n_steps = x.shape[-2]

        k_in = self.k_in(x) # shape (*batch_dims, time, features)
        k_pos = self.k_pos(x) # shape (*batch_dims, time, features)
        q_in = self.q_in(x) # shape (*batch_dims, time, features)
        q_pos = self.q_pos(x) # shape (*batch_dims, time, features)

        # activation
        k_in = nn.sigmoid(k_in)
        k_pos = nn.sigmoid(k_pos)
        q_in = nn.sigmoid(q_in)
        q_pos = nn.sigmoid(q_pos)

        # # Roll the positional features to create a cyclic permutation
        # # Note, we should roll on the feature axis but by an amount equal to the time index
        # # First, rehsape putting time axis first to make it easier to roll
        # k_pos = np.moveaxis(k_pos, -2, 0) # shape (time, *batch_dims, features)
        # q_pos = np.moveaxis(q_pos, -2, 0) # shape (time, *batch_dims, features)
        # # Now roll each time step by its index
        # def f_roll(k_pos_t, t):
        #     # roll on last axis, features, by t
        #     return np.roll(k_pos_t, shift=t, axis=-1)
        # v_f_roll = vmap(f_roll, in_axes=(0, 0))
        # k_pos = v_f_roll(k_pos, np.arange(n_steps)) # shape (time, *batch_dims, features)
        # q_pos = v_f_roll(q_pos, np.arange(n_steps)) # shape (time, *batch_dims, features)
        # # Move time axis back to its original position
        # k_pos = np.moveaxis(k_pos, 0, -2) # shape (*batch_dims, time, features)
        # q_pos = np.moveaxis(q_pos, 0, -2) # shape (*batch_dims, time, features)

        # outer product
        k = k_in[..., :, None] * k_pos[..., None, :] # shape (*batch_dims, time, features, features)
        k = k.reshape((*batch_shape, n_steps, -1)) # shape (*batch_dims, time, features*features)

        q = q_in[..., :, None] * q_pos[..., None, :] # shape (*batch_dims, time, features, features)
        q = q.reshape((*batch_shape, n_steps, -1)) # shape (*batch_dims, time, features*features)

        # Compute linear attention scores using dot product
        s = np.einsum('...qf,...kf->...qk', q, k) # shape (*batch_dims, time, time)
        # Divide by the sum
        s = s / np.sum(s, axis=-1, keepdims=True) # shape (*batch_dims, time, time)
        

        # Compute values
        v = self.v(x) # shape (*batch_dims, time, out_features)
        v = np.einsum('...qk,...kv->...qv', s, v) # shape (*batch_dims, time, out_features)

        out = {
            'k_in': k_in,
            'k_pos': k_pos,
            'q_in': q_in,
            'q_pos': q_pos,
            'v': v,
        }

        aux = {
            "logit": v,
        }

        return [out], aux

class Bonk(nn.Module):
    """
    Test architecture
    """
    
    features: int
    out_features: int

    def setup(self):
        # keys
        self.k = nn.Dense(self.features)
        # queries
        self.q = nn.Dense(self.features)
        # values
        self.v = nn.Dense(self.out_features)

    def __call__(self, x):
        # assume input x is of shape (*batch_dims, time, features)
        batch_shape = x.shape[:-2]
        n_steps = x.shape[-2]

        