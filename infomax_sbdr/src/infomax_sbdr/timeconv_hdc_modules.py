import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable, Tuple
from functools import partial
import infomax_sbdr.binary_comparisons as bc
from math import prod


""" Index-Shift convolution kernels """

def make_shift_conv_kernel(window_size: int, feature_dim: int):
    """
    Returns kernel of shape (W, F, F)
    where each slice is a circulant shift matrix.
    """
    W, F = window_size, feature_dim

    # shifts: (W,)
    shifts = np.arange(W - 1, -1, -1)

    # base indices: (F,)
    idx = np.arange(F)

    # build (W, F, F)
    # kernel[i, out_f, in_f] = 1 if in_f == (out_f - shift) % F
    in_idx = (idx[None, :, None] - shifts[:, None, None]) % F
    out_idx = idx[None, None, :]

    kernel = (in_idx == out_idx).astype(np.float32)

    return kernel


""" Convolutional modules """

class ConvShiftLayer(nn.Module):
    """
    Module with a single dense encoder and a circulant shift convolution layer,
        similary to High-Dimensional Computing (HDC).
    We shift each element in the convolution window depending on the index,
        and sum over the window.
    """
    features: int
    conv_size: int
    conv_stride: int
    padding: str = "SAME"

    def setup(self):
        self.layer = nn.Dense(
            features=self.features,
            kernel_init=jax.nn.initializers.lecun_normal(),
        )

        self.shift_kernel = make_shift_conv_kernel(
            window_size=self.conv_size,
            feature_dim=self.features,
        )

    def __call__(self, x):
        # assume input x is of shape (*batch_dims, time, features)
        batch_shape = x.shape[:-2]

        # encode
        x = self.layer(x)

        # compute bounded pre-activation (tanh)
        a = jax.nn.tanh(x)
        # and binary activation
        z = (a > 0).astype(np.float32)

        # Convolve with Shift Kernel, for next layer
        # This represent a binding operations over vectors in a conv window
        z = z.reshape((-1, *z.shape[-2:]))
        z_conv = jax.lax.conv_general_dilated(
            z.reshape((-1, *z.shape[-2:])),
            self.shift_kernel,
            window_strides=(self.conv_stride,),
            padding=self.padding,
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        z_conv = z_conv.reshape((*batch_shape, *z_conv.shape[-2:]))

        return {
            "a": a,
            "z": z,
            "z_conv": z_conv,
        }

class ConvShift(nn.Module):

    features: Sequence[int]
    conv_sizes: Sequence[Tuple[int]]
    conv_strides: Sequence[Tuple[int, int]]
    padding: str = "SAME"

    def setup(self):
        self.layers = [
            ConvShiftLayer(
                features=self.features[i],
                conv_size=self.conv_sizes[i],
                conv_stride=self.conv_strides[i],
                padding=self.padding
            )
            for i in range(len(self.features))
        ]

    def __call__(self, x):
        
        # return all intermediate outputs
        outs = []

        for layer in self.layers:
            out = layer(x)
            outs.append(out)
            # set input to next layer
            x = jax.lax.stop_gradient(out["z_conv"])

        return outs