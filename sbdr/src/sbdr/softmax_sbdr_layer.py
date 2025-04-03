import jax
import jax.numpy as np
from jax import jit, grad, vmap
import sparse_distributed_memory.binary_comparisons as bn
import flax.linen as nn
from typing import Sequence, Callable
from functools import partial
from flax.core import freeze, unfreeze, FrozenDict
from sparse_distributed_memory.utils import conv1d


class SoftmaxSBDR(nn.Module):
    """
    NN with final output given by n_active_out_features different softmax over different subset of the n_out_features
    each of dimension n_out_features//n_active_out_features
    """

    n_hid_features: Sequence[int]
    n_out_features: int
    n_active_out_features: int
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


class SoftmaxSBDRTimeConv(SoftmaxSBDR):
    """
    Same as the SoftmaxSBDR, but with time convolution before processing the inputs
    """

    n_hid_features: Sequence[int]
    n_out_features: int
    n_active_out_features: int
    activation_fn: Callable = nn.elu
    gamma: float = 0.9
    seq_length: int = 20
    conv_mode: str = "valid"

    def setup(self):

        assert self.n_out_features % self.n_active_out_features == 0
        self.n_group_out_features = self.n_out_features // self.n_active_out_features

        super().setup()

        # compute the decaying weights to be used for convolution
        decaying_weights = np.geomspace(
            self.gamma ** (self.seq_length - 1), 1, self.seq_length
        )
        # normalize and store in an attribute of the class
        self.decaying_weights_back = decaying_weights / np.sum(decaying_weights)
        self.decaying_weights_forw = np.flip(self.decaying_weights_back)

        # define the custom convolution operator
        # we convolve on the second-to-last dimension,
        # assuming input shape is (*batch_dims, time, features)
        self.conv_op_back = partial(
            conv1d,
            w=self.decaying_weights_back,
            axis=-2,
            mode=self.conv_mode,
        )
        self.conv_op_forw = partial(
            conv1d,
            w=self.decaying_weights_forw,
            axis=-2,
            mode=self.conv_mode,
        )

    def forward_conv_back(self, x):
        # input x of shape (*batch_dims, time, features)
        # we convolve on the second-to-last dimension
        x = self.conv_op_back(x)
        x = self(x)
        return x

    def forward_conv_back_align(self, x):
        # input x of shape (*batch_dims, time, features)
        # we convolve on the second-to-last dimension
        # here we return also a cut version of the input that is aligned in time with the convolution
        y = self.conv_op_back(x)
        y = self(y)
        x_aligned = x[..., self.seq_length - 1 :, :]
        return x_aligned, y

    def forward_conv_forw(self, x):
        # input x of shape (*batch_dims, time, features)
        # we convolve on the second-to-last dimension
        x = self.conv_op_forw(x)
        x = self(x)
        return x

    def forward_conv_forw_align(self, x):
        # input x of shape (*batch_dims, time, features)
        # we convolve on the second-to-last dimension
        # here we return also a cut version of the input that is aligned in time with the convolution
        y = self.conv_op_forw(x)
        y = self(y)
        x_aligned = x[..., : -self.seq_length + 1, :]
        return x_aligned, y


class SimplePredictorSBDR(nn.Module):
    """
    Simple NN with final dense output
    """

    n_hid_features: Sequence[int]
    n_out_features: int
    n_active_out_features: int
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
        return x

    def reshaped_softmax(self, x):
        # NOTE: this is a convenience function to be used when we want to apply a softmax
        # to a predict difference of discounted sums, but we want the softmax to be
        # applied applied splitting the output into groups, similar to the Softmax SBDR layers

        # input x of shape (*batch_dims, features)
        # perform softmax on reshaped output
        x = x.reshape(
            x.shape[:-1] + (self.n_active_out_features, self.n_group_out_features)
        )
        x = nn.softmax(x, axis=-1)
        # reshape back
        x = x.reshape(x.shape[:-2] + (self.n_out_features,))
        return x
