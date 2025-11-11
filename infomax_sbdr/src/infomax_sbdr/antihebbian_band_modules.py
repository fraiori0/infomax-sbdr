import jax
import jax.numpy as np
import flax.linen as nn
from functools import partial
from infomax_sbdr.utils import conv1d


""" -------------------- """
""" "Standard" Anti-Hebbian Modules" """
""" -------------------- """


class AntiHebbianBandBase(nn.Module):
    """Base class for a Anti-Hebbian Layer with a single set of weights"""

    n_features: int
    p_target: float
    momentum: float = 0.95

    def setup(self):
        self.wb = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w_forward,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(self.bias_init_value),
        )
        self.b2 = self.param(
            "b2",
            nn.initializers.constant(0.0),
            (self.n_features,),
            np.float32,
        )

        # refs for defining and using "variable" in linen modules
        # https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/state_params.html
        # https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.variable
        # https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/variable.html#module-flax.core.variables

        # running mean/median that tracks (co)activation of units
        # note, the diagonal is simply the running mean of the activation of a single unit,
        # because the activity is either 0 or 1 (so it's not affected by multiplication with itself)
        # mu0 = 0.0
        mu0 = np.ones((self.n_features, self.n_features)) * self.p_target**2 + np.eye(
            self.n_features
        ) * (self.p_target - self.p_target**2)

        self.mu = self.param(
            "mu",
            # nn.initializers.constant(self.p_target),
            nn.initializers.constant(mu0),
            (self.n_features, self.n_features),
            np.float32,
        )

    def initialize_params(self, params, x):

        """ Use this after the real model initi to get unit-norm weights and compute biases depending on initial statistics of some samples """

        # normalize the kernel vectors (unit norm in the input space)
        kernel = params["params"]["wb"]["kernel"]
        kernel = kernel / np.linalg.norm(kernel, axis=1, keepdims=True)

        # Center each band on the mean of projected data
        y = self.w_f(x).reshape(-1, self.n_features)
        b1 = -y.mean(axis=0)

        # compute the 10th percentile of activation for each units, and set that as the value of the bias b2
        # the unit is activated for values lower than that
        b2 = np.quantile(np.abs(y), self.p_target, axis=0)

        # Update the parameter dictionary passed as input
        params["params"]["wb"]["kernel"] = kernel
        params["params"]["wb"]["bias"] = b1
        params["params"]["b2"] = b2

        return params

    def __call__(self, x):
        raise NotImplementedError

    def compute_dmu(self, params, outs, momentum):
        # input z of shape (*batch, features)
        z = outs["z"]

        # matrix of coactivations
        zi_zj = z[..., :, None] * z[..., None, :]

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes

        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - momentum) * (zi_zj - params["params"]["mu"])
        # mean on all batch dimensions
        dmu = dmu.reshape(-1, z.shape[-1], z.shape[-1]).mean(axis=0)

        return dmu

    def compute_dwb(self, params, outs):
        # outs expected to have the same content as returned from the __call__ method of this class

        x = outs["x"]
        z = outs["z"]

        kernel = params["params"]["w_f"]["kernel"]
        mu_self = np.diag(params["params"]["mu"])

        xi_zj = (x[..., :, None] - kernel) * z[..., None, :]

        dkernel = (self.p_target**2 - mu_self**2) * xi_zj
        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        dbias = self.p_target - mu_self

        dw_f = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw_f

    def compute_dw_l(self, params, x, y, z):
        # x and z as returned from the __call__ method

        mu = params["params"]["mu"]
        mu_self = np.diag(params["params"]["mu"])

        kernel = params["params"]["w_l"]["kernel"]

        # matrix of co-active units, used to mask the updates
        # i.e., we update only weights between co-actived units
        zi_zj = z[..., :, None] * z[..., None, :]

        dkernel = (self.p_target**2 - mu) * zi_zj

        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        dbias = self.p_target - mu_self

        dw_l = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw_l

    def compute_dparams(self, params, x, y, z, momentum=None):
        # x and z as returned from the __call__ method
        dparams = {
            "params": {
                "mu": None,
                "w_f": {
                    "kernel": None,
                    "bias": None,
                },
                "w_l": {
                    "kernel": None,
                    "bias": None,
                },
            }
        }

        tmp_params = params.copy()

        if momentum is None:
            dmu = self.compute_dmu(params, z, self.momentum)
        else:
            dmu = self.compute_dmu(params, z, momentum)
        tmp_params["params"]["mu"] = params["params"]["mu"] + dmu

        # we do this because it should be preferable to first update mu,
        # and then use it to update the forward and lateral weights

        dw_f = self.compute_dw_f(tmp_params, x, y, z)
        dw_l = self.compute_dw_l(tmp_params, x, y, z)

        dparams["params"]["mu"] = dmu
        dparams["params"]["w_f"]["kernel"] = dw_f["kernel"]
        dparams["params"]["w_f"]["bias"] = dw_f["bias"]
        dparams["params"]["w_l"]["kernel"] = dw_l["kernel"]
        dparams["params"]["w_l"]["bias"] = dw_l["bias"]

        return dparams

    def apply_dparams(self, params, dparams, lr: float = 0.1):
        # update mu (exponential moving average of feature co-activations)
        params["params"]["mu"] = params["params"]["mu"] + dparams["params"]["mu"]

        # update w_f (feedforward weights)
        params["params"]["w_f"]["kernel"] = (
            params["params"]["w_f"]["kernel"] + lr * dparams["params"]["w_f"]["kernel"]
        )
        params["params"]["w_f"]["bias"] = (
            params["params"]["w_f"]["bias"] + lr * dparams["params"]["w_f"]["bias"]
        )

        # update w_l (lateral weights)
        params["params"]["w_l"]["kernel"] = (
            params["params"]["w_l"]["kernel"] + lr * dparams["params"]["w_l"]["kernel"]
        )
        params["params"]["w_l"]["bias"] = (
            params["params"]["w_l"]["bias"] + lr * dparams["params"]["w_l"]["bias"]
        )

        # NOTE!!
        # IMPORTANT!!! set kernel diagonal to zero
        params["params"]["w_l"]["kernel"] = params["params"]["w_l"]["kernel"] - np.diag(
            np.diag(params["params"]["w_l"]["kernel"])
        )

        return params

    def update_params(self, params, x, y, z, lr: float = 0.1, momentum=None):

        dparams = self.compute_dparams(params, x, y, z, momentum)

        params = self.apply_dparams(params, dparams, lr)

        return params, dparams

        


class AntiHebbianBandModule(AntiHebbianBandBase):
    """Basic version of the Anti-Hebbian Layer, with forward (input) and lateral weights."""
    use_dropout: bool = False
    dropout_rate: float = 0.2
    training: bool = False

    def setup(self):
        # setup the base class
        super().setup()

        # define dropout layer
        if self.use_dropout:
            self.dropout = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)

    def __call__(self, x):
        # input x of shape (*batch, features)
        # i.e., already flattened on the feature dimensions, if multiple (like image C, H, W)

        y = self.w_f(x) > 0
        if self.use_dropout:
            z = self.w_l(self.dropout(y.astype(x.dtype))) > 0
        else:
            z = self.w_l(y) > 0
        z = np.logical_and(y, z)

        return {
            "x": x, 
            "y": y.astype(x.dtype),
            "z": z.astype(x.dtype),
        }
    

