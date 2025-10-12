import jax
import jax.numpy as np
import flax.linen as nn
from functools import partial
from infomax_sbdr.utils import conv1d


""" -------------------- """
""" "Standard" Anti-Hebbian Modules" """
""" -------------------- """


class AntiHebbianXORBase(nn.Module):
    """Base class for a Anti-Hebbian Layer with XOR activation, with forward (input) and lateral weights."""

    n_features: int
    p_target: float
    momentum: float = 0.95
    init_variance_w_forward: float = 1.0
    init_variance_w_lateral: float = 1.0
    bias_init_value: float = 0.0

    def setup(self):
        self.w_f = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w_forward,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(self.bias_init_value),
        )
        self.w_l = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w_lateral,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(self.bias_init_value),
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

    def __call__(self, x):
        raise NotImplementedError

    def compute_dmu(self, params, z, momentum):
        # input z of shape (*batch, features)

        # matrix of coactivations
        zi_zj = z[..., :, None] * z[..., None, :]

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes

        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - momentum) * (zi_zj - params["params"]["mu"])
        # mean on all batch dimensions
        dmu = dmu.reshape(-1, z.shape[-1], z.shape[-1]).mean(axis=0)

        return dmu

    def compute_dw_f(self, params, x, y, u, z):
        # x and z as returned from the __call__ method

        kernel = params["params"]["w_f"]["kernel"]
        bias = params["params"]["w_f"]["bias"]
        mu_self = np.diag(params["params"]["mu"])

        xi_zj = (x[..., :, None] - kernel) * z[..., None, :]
        dkernel = (self.p_target**2 - mu_self**2) * xi_zj * 2 * (0.5 - u[..., None, :])
        dbias = (self.p_target - mu_self) * 2 * (0.5 - u)

        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        dbias = dbias.reshape(-1, *bias.shape).mean(axis=0)

        dw_f = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw_f

    def compute_dw_l(self, params, x, y, u, z):
        # x and z as returned from the __call__ method

        mu = params["params"]["mu"]
        mu_self = np.diag(params["params"]["mu"])

        kernel = params["params"]["w_l"]["kernel"]
        bias = params["params"]["w_l"]["bias"]

        # matrix of co-active units, used to mask the updates
        # i.e., we update only weights between co-actived units
        yi_zj = y[..., :, None] * z[..., None, :]
        # yi_zj = (y[..., :, None] - kernel) * z[..., None, :]
        dkernel = (self.p_target**2 - mu) * yi_zj * 2 * (0.5 - y[..., None, :])
        dbias = (self.p_target - mu_self) * 2 * (0.5 - y)

        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        dbias = dbias.reshape(-1, *bias.shape).mean(axis=0)

        dw_l = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw_l

    def compute_dparams(self, params, x, y, u, z, momentum=None):
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

        dw_f = self.compute_dw_f(tmp_params, x, y, u, z)
        dw_l = self.compute_dw_l(tmp_params, x, y, u, z)

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

    def update_params(self, params, x, y, u, z, lr: float = 0.1, momentum=None):

        dparams = self.compute_dparams(params, x, y, u, z, momentum)

        params = self.apply_dparams(params, dparams, lr)

        return params, dparams


class AntiHebbianXORModule(AntiHebbianXORBase):
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

        if self.use_dropout:
            y = self.w_f(self.dropout(x)) > 0
        else:
            y = self.w_f(x) > 0
        u = self.w_l(y) > 0

        z = np.logical_xor(y, u)

        return {
            "x": x, 
            "y": y.astype(x.dtype),
            "u": u.astype(x.dtype),
            "z": z.astype(x.dtype),
        }
    

