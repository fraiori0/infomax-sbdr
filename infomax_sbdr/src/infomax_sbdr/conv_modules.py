import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable, Tuple
from functools import partial
import infomax_sbdr.binary_comparisons as bc


""" Custom Pool functions"""


def or_pool(inputs, window_shape, strides=None, padding="VALID"):
    y = 1.0 - inputs
    y = nn.pool(y, 1.0, jax.lax.mul, window_shape, strides, padding)
    return 1.0 - y


def and_pool(inputs, window_shape, strides=None, padding="VALID"):
    y = nn.pool(y, 1.0, jax.lax.mul, window_shape, strides, padding)
    return y


""" Convolutional modules """


class ConvBase(nn.Module):
    """
    Model with a sequence of convolutional layers, without pooling.
    """

    kernel_features: Sequence[int] = None
    kernel_sizes: Sequence[Tuple[int]] = None
    kernel_strides: Sequence[Tuple[int, int]] = None
    kernel_padding: str = "SAME"
    pool_fn: Callable = None
    pool_sizes: Sequence[Tuple[int, int]] = None
    pool_strides: Sequence[Tuple[int, int]] = None
    activation_fn: Callable = nn.leaky_relu

    def __call__(self, x):
        # input x of shape (*batch_dims, channel, height, width)
        x = self.layers(x)
        return x


class ConvNoPoolBase(ConvBase):
    """
    Model with a sequence of convolutional layers, without pooling.
    """

    def setup(self):
        assert len(self.kernel_features) == len(self.kernel_sizes)
        assert len(self.kernel_sizes) == len(self.kernel_strides)

        layers = []
        for f, k, s in zip(
            self.kernel_features, self.kernel_sizes, self.kernel_strides
        ):
            layers.append(
                nn.Conv(
                    features=f,
                    kernel_size=k,
                    strides=s,
                    padding=self.kernel_padding,
                ),
            )
            layers.append(self.activation_fn)
        # create the callable applying all layers in sequence
        self.layers = nn.Sequential(layers)


class ConvNoPoolNoLastBase(ConvBase):
    """
    Model with a sequence of convolutional layers, without pooling.
    Note, last layer does not have an activation
    """

    def setup(self):
        assert len(self.kernel_features) == len(self.kernel_sizes)
        assert len(self.kernel_sizes) == len(self.kernel_strides)

        layers = []
        for f, k, s in zip(
            self.kernel_features, self.kernel_sizes, self.kernel_strides
        ):
            layers.append(
                nn.Conv(
                    features=f,
                    kernel_size=k,
                    strides=s,
                    padding=self.kernel_padding,
                ),
            )
            layers.append(self.activation_fn)
        # remove last activation
        layers = layers[:-1]
        # create the callable applying all layers in sequence
        self.layers = nn.Sequential(layers)


class ConvFLONoPoolNoLast(ConvNoPoolNoLastBase):

    def setup(self):

        super().setup()

        # add a final layer returning the negPMI
        self.neg_pmi_layer = nn.Dense(features=1)

    def __call__(self, x):
        x = self.layers(x)
        # flatten x
        x = x.reshape((*x.shape[:-3], -1))
        # sigmoid
        x = nn.sigmoid(x)
        # negPMI
        negpmi = self.neg_pmi_layer(x)
        return x, negpmi


class ConvFLONoPool(ConvNoPoolBase):
    output_features: int = None

    def setup(self):

        super().setup()
        # add a final dense layer
        self.final_dense = nn.Dense(features=self.output_features)
        # add a final layer returning the negPMI
        self.neg_pmi_layer = nn.Dense(features=1)

    def __call__(self, x):
        x = self.layers(x)
        # flatten x
        x = x.reshape((*x.shape[:-3], -1))
        # apply final dense layer
        x = self.final_dense(x)
        # sigmoid
        x = nn.sigmoid(x)
        # negPMI
        negpmi = self.neg_pmi_layer(x)
        return x, negpmi
