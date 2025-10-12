import flax.linen as nn
import optax
from infomax_sbdr.dense_modules import *
from infomax_sbdr.conv_modules import *
from infomax_sbdr.transforms import *
import infomax_sbdr.binary_comparisons as bc
from torch import nn as torch_nn
import infomax_sbdr.classifier_modules as classifier_modules
import infomax_sbdr.antihebbian_modules as ah
import infomax_sbdr.antihebbian_recurrent_modules as rec_ah
import infomax_sbdr.antihebbian_xor_modules as xor_ah


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
    "identity": lambda x: x,
    "mish": jax.nn.mish,
}

config_torch_activation_dict = {
    "relu": torch_nn.ReLU(),
    "leaky_relu": torch_nn.LeakyReLU(),
    "tanh": torch_nn.Tanh(),
    "sigmoid": torch_nn.Sigmoid(),
    "softmax": torch_nn.Softmax(dim=-1),
    "elu": torch_nn.ELU(),
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
    "DenseFLO": DenseFLO,
    "ConvFLONoPoolNoLast": ConvFLONoPoolNoLast,
    "ConvFLONoPool": ConvFLONoPool,
    "VGGFLO": VGGFLO,
    "VGGFLOAutoEncoder": VGGFLOAutoEncoder,
    "VGGFLOKSoftMax": VGGFLOKSoftMax,
    "VGGGlobalPoolFLO": VGGGlobalPoolFLO,
    "VGGFLOMultiLayerNEGPMI": VGGFLOMultiLayerNEGPMI,
    "VGGDecoder": VGGDecoder,
    "VGGGlobalPoolFLOMultiLayerNEGPMI": VGGGlobalPoolFLOMultiLayerNEGPMI,
}

config_ah_module_dict = {
    "AntiHebbianModule": ah.AntiHebbianModule,
    "AntiHebbianRecurrentModule": rec_ah.AntiHebbianRecurrentModule,
    "AntiHebbianXORModule": xor_ah.AntiHebbianXORModule
}


config_classifier_module_dict = {
    "DenseClassifier": classifier_modules.DenseClassifier,
}

config_transform_dict = {
    "minmax": minmax_transform,
    "offsetscale": offsetscale_transform,
}

config_similarity_dict = {
    "jaccard": bc.jaccard_index,
    "and": bc.expected_and,
    "cosine_normalized": bc.cosine_similarity_normalized,
    "asym_jaccard": bc.asymmetric_jaccard_index,
    "log_and": bc.log_and,
}
