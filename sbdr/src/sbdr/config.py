import flax.linen as nn
import optax
from sbdr.modules import *
from sbdr.transforms import *

"""
Convenience dictionaries to match strings in config files to functions and classes.
"""

config_activation_dict = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "softmax": nn.softmax,
    "elu": nn.elu,
    "selu": nn.selu,
    "gelu": nn.gelu,
    "leaky_relu": nn.leaky_relu,
}

config_optimizer_dict = {
    "sgd": optax.sgd,
    "adam": optax.adam,
    "adamw": optax.adamw,
    "adagrad": optax.adagrad,
    "rmsprop": optax.rmsprop,
    "adadelta": optax.adadelta,
    "adamax": optax.adamax,
}

config_module_dict = {
    "DenseFLOSigmoid": DenseFLOSigmoid,
}

config_transform_dict = {
    "minmax": minmax_transform,
    "offsetscale": offsetscale_transform,
}
