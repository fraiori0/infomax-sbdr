import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable
from functools import partial


class DenseFLO(nn.Module):
    """
    MLP computing also a neg-pmi score
    """

    hid_features: Sequence[int]
    out_features: int
    hid_features_negpmi: Sequence[int]
    activation_fn: Callable = jax.nn.mish
    out_activation_fn: Callable = nn.sigmoid
    use_batchnorm: bool = False
    use_dropout: bool = False
    dropout_rate: float = 0.0
    training: bool = True  # whether in training or eval mode (for dropout/batchnorm)

    def setup(self):

        layers = []
        for f in self.hid_features:
            # dense layer
            layers.append(nn.Dense(features=f))
            # batch norm
            if self.use_batchnorm:
                layers.append(nn.BatchNorm(use_running_average=not self.training))
            # activation
            layers.append(self.activation_fn)
            # dropout
            if self.use_dropout:
                layers.append(nn.Dropout(rate=self.dropout_rate, deterministic=not self.training))

        layers.append(nn.Dense(features=self.out_features))
        layers.append(self.out_activation_fn)
        self.layers = nn.Sequential(layers)

        negpmi_layers = []
        for f in self.hid_features_negpmi:
            negpmi_layers.append(nn.Dense(features=f))
            negpmi_layers.append(self.activation_fn)
        negpmi_layers.append(nn.Dense(features=1))
        self.negpmi_layers = nn.Sequential(negpmi_layers)

    def __call__(self, x):
        # input x of shape (*batch_dims, features)
        # i.e., already flattened on the feature dimensions,
        # if there were multiple in the original data (like image C, H, W)

        # apply the feedforward weights
        x = self.layers(x)
        # compute negpmi
        negpmi = self.negpmi_layers(x)
        
        return {
            "z": x,
            "neg_pmi": negpmi
        }