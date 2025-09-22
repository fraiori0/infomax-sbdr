import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable
from functools import partial


class DenseClassifier(nn.Module):
    """
    Standard feedforward neural network classifier with a final linear layer
    The final output is suppose to be softmax-ed afterwards.
    """

    out_features: int
    dense_features: Sequence[int]
    activation_fn: Callable = nn.elu

    def setup(self):
        layers = []
        for f in self.dense_features:
            layers.append(nn.Dense(features=f))
            layers.append(self.activation_fn)
        layers.append(nn.Dense(features=self.out_features))
        self.layers = nn.Sequential(layers)

    def __call__(self, x):
        # apply the feedforward weights
        x = self.layers(x)
        # note, a softmax is supposed to be applied later
        return {"logits": x}
