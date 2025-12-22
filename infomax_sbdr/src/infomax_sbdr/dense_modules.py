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
    

class HyperplaneLayer(nn.Module):
    """
    Layer where each unit is a hyperplane and the output is the distance from the hyperplane.
    """
    out_features: int

    def setup(self):
        self.h = nn.Dense(features=self.out_features)
        

    def __call__(self, x):
        # for initialization
        _ = self.h(x)

        kernel = self.variables["params"]["h"]["kernel"]
        bias = self.variables["params"]["h"]["bias"]

        # normalize weights to unit norm (separately for each unit)
        kernel = kernel / (np.linalg.norm(kernel, axis=-2, keepdims=True) + 1e-6)

        # compute the squared distances from the hyperplane w^T x + b = 0
        d = (x[..., None] * kernel).sum(axis=-2) + bias
        d = d**2

        return {"d": d}
    
class HyperplaneSlabLayer(HyperplaneLayer):
    """
    Layer where the activation is a an activation applied to the distance from an hyperplane+a bias parameter
    """
    activation_fn : Callable = nn.sigmoid

    def setup(self):
        super().setup()
        self.b = self.param(
            "b",
            nn.initializers.constant(0.0),
            (self.out_features,),
            np.float32
        )
    
    def __call__(self, x):
        outs = super().__call__(x)
        outs["z"] = self.activation_fn(-outs["d"] + self.b)
        return outs
    
class DenseSlabFLO(nn.Module):
    """
    Dense network with final HyperplaneSlabLayer and negpmi
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

        # Append a final HyperplaneSlabLayer
        layers.append(
            HyperplaneSlabLayer(
                out_features=self.out_features,
                activation_fn=self.out_activation_fn,
            )
        )

        self.layers = nn.Sequential(layers)

        negpmi_layers = []
        for f in self.hid_features_negpmi:
            negpmi_layers.append(nn.Dense(features=f))
            negpmi_layers.append(self.activation_fn)
        negpmi_layers.append(nn.Dense(features=1))
        self.negpmi_layers = nn.Sequential(negpmi_layers)

    def __call__(self, x):
        outs = self.layers(x)
        outs["neg_pmi"] = self.negpmi_layers(outs["z"])
        return outs
    
class ScaledRBF(nn.Module):
    n_units : int
    in_features : int
    def setup(self,):
        
        # Scale
        self.s = nn.Dense(
            features=self.n_units,
            use_bias=False,
            kernel_init=nn.initializers.ones,
        )

        # Centroid parameters
        self.c = nn.Dense(
            features=self.n_units,
            use_bias=True,
            kernel_init=nn.initializers.lecun_normal(),
        )

    def __call__(self, x):

        # for initialization
        _ = self.s(x)
        _ = self.c(x)

        s = self.variables["params"]["s"]["kernel"]
        s = jax.nn.softplus(s)

        c = self.variables["params"]["c"]["kernel"]

        # compute distances in scaled space
        diff = (x[..., None] - c)
        diff = diff * s
        # use mean instead of sum to account for the distance scale given by the dimensions of the data
        d_sq = np.mean(diff**2, axis=-2)
        

        # activate rbf centroids
        z = np.exp(-d_sq)

        return {"z": z}

class ScaledRBFFLO(ScaledRBF):
    hid_features_negpmi: Sequence[int]
    activation_fn: Callable = jax.nn.mish
    out_activation_fn: Callable = nn.sigmoid
    use_batchnorm: bool = False
    use_dropout: bool = False
    dropout_rate: float = 0.0
    training: bool = True  # whether in training or eval mode (for dropout/batchnorm)

    def setup(self):
        super().setup()

        negpmi_layers = []
        for f in self.hid_features_negpmi:
            negpmi_layers.append(nn.Dense(features=f))
            negpmi_layers.append(self.activation_fn)
        negpmi_layers.append(nn.Dense(features=1))
        self.negpmi_layers = nn.Sequential(negpmi_layers)

    def __call__(self, x):
        outs = super().__call__(x)
        outs["neg_pmi"] = self.negpmi_layers(outs["z"])
        return outs
