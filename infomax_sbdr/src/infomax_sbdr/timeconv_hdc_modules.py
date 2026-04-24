import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable, Tuple
from functools import partial
import infomax_sbdr.binary_comparisons as bc
import infomax_sbdr.utils as ut


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


# standard Temporal Convolutional Network
class TemporalConvLayer(nn.Module):
    """Causal temporal convolutional layer.
 
    Attributes:
        features:      Number of output channels (kernel features).
        kernel_size:   Length of the 1-D convolutional kernel (must be ≥ 1).
        stride:        Step size along the time axis (default 1).
        use_bias:      Whether to add a learnable bias term.
        kernel_init:   Initialiser for the convolutional kernel weights.
        bias_init:     Initialiser for the bias vector.
    """
 
    features: int
    kernel_size: int
    stride: int = 1
    use_bias: bool = True
 
    # ------------------------------------------------------------------
    # setup — create sub-modules and validate hyper-parameters
    # ------------------------------------------------------------------
 
    def setup(self) -> None:
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be ≥ 1, got {self.kernel_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {self.stride}")
 
        # We manage causal padding ourselves, so the underlying Conv uses
        # 'VALID' (no implicit padding).
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding="VALID",
            use_bias=self.use_bias,
        )
 
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal temporal convolution.
 
        Args:
            x: Float array of shape ``(*batch_dims, time, features)``.
               Any number of leading batch dimensions is supported.
 
        Returns:
            Float array of shape ``(*batch_dims, time_out, self.features)``
            where ``time_out = ⌊(time − 1) / stride⌋ + 1``.
        """

        th = 0.3

        # 1. Causal padding
        # Prepend (kernel_size − 1) zero frames so each output step only
        # sees the present and past inputs, never future ones.
        #
        # jnp.pad expects a sequence of (before, after) pairs, one per axis.
        # Layout: [...batch axes..., time axis, feature axis]
        pad_size = self.kernel_size - 1
        if pad_size > 0:
            pad_width = [(0, 0)] * (x.ndim - 2) + [(pad_size, 0), (0, 0)]
            x = np.pad(x, pad_width)
 
        # 2. Convolution (VALID — no further padding)
        pre_activation = self.conv(x)
        z = jax.nn.sigmoid(pre_activation)
        y = ut.threshold_softgradient(pre_activation)

        outs = {
            "z": z,
            "y": y,
        }
        return outs


class TCN(nn.Module):
    """Temporal Convolutional Network (TCN) with multiple layers.
    """
 
    features: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int] = 1
    use_bias: bool = True
 
    def setup(self) -> None:
        if len(self.features) != len(self.kernel_sizes):
            raise ValueError("features and kernel_sizes must have the same length")
        if isinstance(self.strides, int):
            self.strides = [self.strides] * len(self.features)
        elif len(self.strides) != len(self.features):
            raise ValueError("strides must be an int or have the same length as features")
 
        self.layers = [
            TemporalConvLayer(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                use_bias=self.use_bias,
            )
            for i in range(len(self.features))
        ]

    def __call__(self, x):
        th = 0.3

        # return all intermediate outputs
        outs = []
        x_in = x
        for layer in self.layers:
            out = layer(x_in)
            outs.append(out)
            # input to next layer
            x_in = jax.lax.stop_gradient(out["y"])
            

        return outs