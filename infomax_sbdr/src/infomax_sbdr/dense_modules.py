import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable
from functools import partial
from infomax_sbdr.utils import threshold_softgradient
import infomax_sbdr.initializers as my_inits


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
        # layers.append(self.out_activation_fn)
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
        a = self.layers(x)
        a = jax.nn.tanh(a)
        # apply output activation function
        z = (a > 0).astype(np.float32)
        # compute negpmi
        negpmi = self.negpmi_layers(a)
        
        return {
            "a": a,
            "z": z,
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


class DenseNCEWeights(nn.Module):
    """
    Single-layer MLP that is supposed to be train by using
    an InfoNCE loss that directly involves the weights.
    Note, weights and inputs should be non-negative to avoid degeneracies
    """

    out_features: int
    in_features: int

    def setup(self):

        # make W and b params. Note, W should be non-negative
        self.w = self.param(
            "w",
            # should be initialized with all non-negative weights
            # using the proper initializers
            init_fn=my_inits.non_negative(scale=0.6),
            shape=(self.in_features, self.out_features),
            dtype=np.float32,
        )

        self.b = self.param(
            "b",
            nn.initializers.constant(0.0),
            (self.out_features,),
            np.float32
        )

    def __call__(self, x):
        # input x of shape (*batch_dims, features)
        # i.e., already flattened on the feature dimensions,
        # if there were multiple in the original data (like image C, H, W)

        # apply the feedforward weights
        a = x @ self.w 
        # no gradient on the weights, they will appear directly in the loss
        z = jax.nn.sigmoid(jax.lax.stop_gradient(a) + self.b)
        
        return {
            "a": a,
            "z": z,
        }

class SparseDictionary(nn.Module):
    """
    Encoder module with also a set of reconstruction weights
    """

    in_features : int
    features : int
    pre_features : Sequence[int]
    n_steps : int
    non_negative_init : bool = False # whether to initialize with a non-negative dictionary
    activation_fn : Callable = jax.nn.gelu
    use_dropout: bool = False
    dropout_rate: float = 0.0
    training: bool = True
    extra_unit_softmax : int = 1 # number of extra units to add to the softmax to allow for empty softmax

    def setup(self):

        assert self.n_steps > 0, "n_steps must be greater than 0"

        # # pre-layers
        # pre_layers = []
        # for i in range(len(self.pre_features)):
        #     pre_layers.append(nn.Dense(features=self.pre_features[i]))
        #     pre_layers.append(self.activation_fn)
        #     # dropout
        #     if self.use_dropout:
        #         pre_layers.append(nn.Dropout(rate=self.dropout_rate, deterministic=not self.training))
                
        # # Stack together
        # self.pre_layers = nn.Sequential(pre_layers)

        # Iterative layer
        self.r = nn.Sequential([
            # nn.Dense(features=self.features),
            # self.activation_fn,
            # note, we use some extra "sink features" to allow for empty softmax
            nn.Dense(features=(self.in_features + self.extra_unit_softmax)), 
        ])

        # Dictionar of atoms, practically a linear layer
        if self.non_negative_init:
            k_init = my_inits.non_negative(scale=1.0/np.sqrt(self.in_features))
        else:
            k_init = nn.initializers.normal(scale=1.0/np.sqrt(self.in_features))
        self.D = nn.Dense(
            features=self.in_features,
            kernel_init=k_init,
        )

    def __call__(self, x):
        # apply pre-layers
        # _ = self.pre_layers(x)
        
        # iterative updates
        def step(crr, inp):
            x_res = crr
            # encode
            z = jax.nn.softmax(self.r(x_res), axis=-1)
            # remove the extra features
            z = z[..., :-self.extra_unit_softmax]
            # reconstruct
            x_hat = self.D(z)
            # remove the reconstruction, work like an iterative residual model
            x_res = x_res - x_hat
            next_crr = x_res
            out = {
                "x_hat": x_hat,
                "z": z
            }
            return next_crr, out
        
        # call the step once, to avoid leaked trace during compilation of jax
        _ = step(x, None)
        
        # scan over the steps
        _, outs_seq = jax.lax.scan(
            step,
            x,
            None,
            length=self.n_steps
        )

        # take the maximum of activations over the steps
        outs_seq["z"] = outs_seq["z"].max(0)
        # sum the reconstruction
        outs_seq["x_hat"] = outs_seq["x_hat"].sum(0)
        
        outs = {
            "x_hat": outs_seq["x_hat"],
            "z": outs_seq["z"],
            "x_original": x,
        }
       
        return outs

        
class SparseDictionaryClassifier(SparseDictionary):
    """
    SparseDictionary with a classifier head
    """

    n_classes : int = 1
    stop_grad_class : bool = True

    def setup(self):
        super().setup()
        self.classifier = nn.Dense(features=self.n_classes)

    def __call__(self, x):
        outs = super().__call__(x)
        # classify using the activations
        c_in = outs["z"]
        if self.stop_grad_class:
            c_in = jax.lax.stop_gradient(c_in)
        logits = self.classifier(c_in)
        outs["logits"] = logits
        return outs



class SparseDenseLayer(nn.Module):
    features: int
    out_features: int
    kernel_init: Callable = nn.initializers.lecun_normal()

    def setup(self):
        self.r = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.constant(-2.0),
        )
        self.d = nn.Dense(
            features=self.out_features,
            kernel_init=self.kernel_init,
            use_bias=False,
        )

    def __call__(self, x):
        z = self.r(x)
        z = jax.nn.sigmoid(z)
        y = self.d(z)
        out = {
            "z": z,
            "y": y,
        }
        return out
    
    