import jax
import jax.numpy as np
from jax import jit, grad, vmap
import sparse_distributed_memory.binary_comparisons as bn
import flax.linen as nn
from typing import Sequence, Callable
from functools import partial
from flax.core import freeze, unfreeze, FrozenDict


# convolution operator over one specified axis, useful for time convolution with arbitrary batch dimensions
def conv1d(x, w, axis: int, mode="valid"):
    return np.apply_along_axis(lambda x: np.convolve(x, w, mode=mode), -2, x)


class FoldiakLayer(nn.Module):
    """Layer that implements feedforward connection and lateral inhibition, with thresholding.
    Note, gradient descent will not work with this layer, as it is not differentiable.
    """

    n_features: int
    p_target: float
    momentum: float = 0.95
    init_variance_q: float = 1.0
    init_variance_w: float = 1.0

    def setup(self):
        self.q = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_q,
                mode="fan_in",
                distribution="truncated_normal",
            ),
        )
        self.w = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w,
                mode="fan_in",
                distribution="truncated_normal",
            ),
        )

        # refs for defining and using "variable" in linen modules
        # https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/state_params.html
        # https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.variable
        # https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/variable.html#module-flax.core.variables

        # running mean/median that tracks (co)activation of units
        # note, the diagonal is simply the running mean of the activation of a single unit,
        # because the activity is either 0 or 1 (so it's not affected by multiplication with itself)
        self.mu = self.param(
            "mu",
            nn.initializers.constant(self.p_target),
            (self.n_features, self.n_features),
            np.float32,
        )

    def __call__(self, x):
        # input x of shape (*batch, features)
        # i.e., already flattened on the feature dimensions, if multiple (like image C, H, W)

        y = self.q(x) > 0
        z = self.w(y) > 0
        z = y * z

        return z

    def compute_dmu(self, params, z):
        n_batch_dims = z.ndim - 1

        # matrix of coactivations
        zi_zj = z[..., :, None] * z[..., None, :]

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes
        dmu = (1.0 - self.momentum) * \
            (zi_zj -
             jax.lax.broadcast(params["params"]["mu"], [1]*n_batch_dims))
        dmu = dmu.mean(axis=0)

        return dmu

    def compute_dq(self, params, x, z):
        # input x of shape (*batch, features)
        # input z of shape (*batch, features)

        q_kernel = params["params"]["q"]["kernel"]
        q_bias = params["params"]["q"]["bias"]
        mu_self = np.diag(params["params"]["mu"])

        n_batch_dims = len(x.shape) - 1

        # # without weight decay
        # xi_zj = x[:, :, None] * z[:, None, :]
        # xi_zj = xi_zj.mean(axis=0)

        # with weight decay
        # (note: not weight, we can interpret it as pulling the weights towards the value of x, like an error term)
        # indeed, if the weights are equal to x, xi_zj = 0, and the weights are not updated
        xi_zj = (x[:, :, None] - q_kernel[None, :, :]) * z[:, None, :]
        # xi_zj = xi_zj.mean(axis=0)

        dq = {
            "kernel": ((self.p_target**2 - mu_self**2)[None, None, :] * xi_zj).mean(
                axis=0
            ),
            "bias": (self.p_target - mu_self),
        }

        return dq

    def compute_dw(self, params, z):

        mu = params["params"]["mu"]
        mu_self = np.diag(params["params"]["mu"])

        w_kernel = params["params"]["w"]["kernel"]

        zi_zj = (z[:, :, None] - 0.0 * w_kernel[None, :, :]) * z[:, None, :]
        # zi_zj = zi_zj.mean(axis=0)

        dw = {
            "kernel": ((self.p_target**2 - mu)[None, :, :] * zi_zj).mean(axis=0),
            "bias": (self.p_target - mu_self),
        }

        return dw

    def compute_dparams(self, params, x, z):
        x = x.reshape((x.shape[0], -1))

        dparams = {
            "params": {
                "mu": None,
                "q": {
                    "kernel": None,
                    "bias": None,
                },
                "w": {
                    "kernel": None,
                    "bias": None,
                },
            }
        }

        params = unfreeze(params.copy())

        dmu = self.compute_dmu(params, z)
        params["params"]["mu"] = params["params"]["mu"] + dmu

        # we do this because it should be preferable to update mu and then use it
        # to compute the update for q and w

        dq = self.compute_dq(params, x, z)
        dw = self.compute_dw(params, z)

        dparams["params"]["mu"] = dmu
        dparams["params"]["q"]["kernel"] = dq["kernel"]
        dparams["params"]["q"]["bias"] = dq["bias"]
        dparams["params"]["w"]["kernel"] = dw["kernel"]
        dparams["params"]["w"]["bias"] = dw["bias"]

        return freeze(dparams)

    def update_params(self, params, x, z, lr: float = 0.1):

        dparams = self.compute_dparams(params, x, z)

        params = params

        params["params"]["mu"] = params["params"]["mu"] + \
            dparams["params"]["mu"]
        params["params"]["q"]["kernel"] = (
            params["params"]["q"]["kernel"] + lr *
            dparams["params"]["q"]["kernel"]
        )
        params["params"]["q"]["bias"] = (
            params["params"]["q"]["bias"] + lr * dparams["params"]["q"]["bias"]
        )
        params["params"]["w"]["kernel"] = (
            params["params"]["w"]["kernel"] + lr *
            dparams["params"]["w"]["kernel"]
        )
        # important, set kernel diagonal to zero
        params["params"]["w"]["kernel"] = params["params"]["w"]["kernel"] - np.diag(
            np.diag(params["params"]["w"]["kernel"])
        )
        params["params"]["w"]["bias"] = (
            params["params"]["w"]["bias"] + lr * dparams["params"]["w"]["bias"]
        )

        return params


class FoldiakTimeConvLayer(nn.Module):
    """Layer that implements feedforward connection and lateral inhibition, with thresholding.
    Note, gradient descent will not work with this layer, as it is not differentiable.

    This layer also assume to have a second-to-last dimension corresponding to time.
    It will convolve over time using a set of weights, computing a backward discounted sum.
    """

    n_features: int
    p_target: float
    momentum: float = 0.95
    init_variance_q: float = 1.0
    init_variance_w: float = 1.0
    gamma: float = 0.9
    seq_length: int = 40
    conv_mode: str = "valid"

    def setup(self):

        self.q = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_q,
                mode="fan_in",
                distribution="truncated_normal",
            ),
        )
        self.w = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w,
                mode="fan_in",
                distribution="truncated_normal",
            ),
        )

        # refs for defining and using "variable" in linen modules
        # https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/state_params.html
        # https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.variable
        # https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/variable.html#module-flax.core.variables

        # running mean/median that tracks (co)activation of units
        # note, the diagonal is simply the running mean of the activation of a single unit,
        # because the activity is either 0 or 1 (so it's not affected by multiplication with itself)
        self.mu = self.param(
            "mu",
            nn.initializers.constant(self.p_target),
            (self.n_features, self.n_features),
            np.float32,
        )

        # compute the decaying weights to be used for convolution
        decaying_weights = np.geomspace(
            self.gamma ** (self.seq_length - 1), 1, self.seq_length
        )
        # normalize and store in an attribute of the class
        self.decaying_weights = decaying_weights / np.sum(decaying_weights)

        # define the custom convolution operator
        # we convolve on the second-to-last dimension,
        # assuming input shape is (batch, time, features)
        self.conv_op = partial(
            conv1d,
            w=self.decaying_weights,
            axis=-2,
            mode=self.conv_mode,
        )

    def __call__(self, x):
        # input x of shape (batch, time, *features)
        x = x.reshape((x.shape[0], x.shape[1], -1))

        # apply the convolution operator
        x = self.conv_op(x)

        # apply the feedforward connection
        y = self.q(x) > 0

        # should we convolve over time here as well?
        # apply the lateral connections
        z = self.w(y) > 0

        # activate only if both y and z are active
        z = y * z

        # return also the convolved x, because the time dimension was modified
        # and we want to use the new one for weight update
        return x, z

    def compute_dmu(self, params, z):
        # input z assumed to be of shape (batch, time, features)
        # with time we added two "None" dimensions when subtracting the current value of mu
        # and take the mean over the first two axis

        # matrix of coactivations
        zi_zj = z[..., :, None] * z[..., None, :]

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes
        dmu = (1.0 - self.momentum) * \
            (zi_zj - params["params"]["mu"][None, None, :, :])
        dmu = dmu.mean(axis=(0, 1))

        return dmu

    def compute_dq(self, params, x, z):
        # input x of shape (batch, time, *features)
        # NOTE it has to be the one returned by the __call__ method tobe time-aligned with z

        # !!! so we don't even reshape it here !!!

        # z of shape (batch, time, features)

        q_kernel = params["params"]["q"]["kernel"]
        q_bias = params["params"]["q"]["bias"]
        mu_self = np.diag(params["params"]["mu"])

        # with weight decay
        xi_zj = (x[:, :, :, None] - q_kernel[None, None, :, :]) * \
            z[:, :, None, :]

        d_kernel = (self.p_target**2 - mu_self**2)[None, None, None, :] * xi_zj
        d_kernel = d_kernel.mean(axis=(0, 1))
        d_bias = self.p_target - mu_self

        dq = {
            "kernel": d_kernel,
            "bias": d_bias,
        }

        return dq

    def compute_dw(self, params, z):

        mu = params["params"]["mu"]
        mu_self = np.diag(params["params"]["mu"])

        w_kernel = params["params"]["w"]["kernel"]

        zi_zj = z[:, :, None] * z[:, :, None, :]
        d_kernel = (self.p_target**2 - mu)[None, None, :, :] * zi_zj
        d_kernel = d_kernel.mean(axis=(0, 1))
        d_bias = self.p_target - mu_self

        dw = {
            "kernel": d_kernel,
            "bias": d_bias,
        }

        return dw

    def compute_dparams(self, params, x, z):
        # input x of shape (batch, time, *features)
        # NOTE it has to be the one returned by the __call__ method tobe time-aligned with z

        # !!! so we don't even reshape it here !!!

        # z of shape (batch, time, features)

        dparams = {
            "params": {
                "mu": None,
                "q": {
                    "kernel": None,
                    "bias": None,
                },
                "w": {
                    "kernel": None,
                    "bias": None,
                },
            }
        }

        params = unfreeze(params.copy())

        dmu = self.compute_dmu(params, z)
        params["params"]["mu"] = params["params"]["mu"] + dmu

        # we do this because it should be preferable to update mu and then use it
        # to compute the update for q and w

        dq = self.compute_dq(params, x, z)
        dw = self.compute_dw(params, z)

        dparams["params"]["mu"] = dmu
        dparams["params"]["q"]["kernel"] = dq["kernel"]
        dparams["params"]["q"]["bias"] = dq["bias"]
        dparams["params"]["w"]["kernel"] = dw["kernel"]
        dparams["params"]["w"]["bias"] = dw["bias"]

        return dparams

    def apply_dparams(self, params, dparams, lr: float = 0.1):

        params["params"]["mu"] = params["params"]["mu"] + \
            dparams["params"]["mu"]
        params["params"]["q"]["kernel"] = (
            params["params"]["q"]["kernel"] + lr *
            dparams["params"]["q"]["kernel"]
        )
        params["params"]["q"]["bias"] = (
            params["params"]["q"]["bias"] + lr * dparams["params"]["q"]["bias"]
        )
        params["params"]["w"]["kernel"] = (
            params["params"]["w"]["kernel"] + lr *
            dparams["params"]["w"]["kernel"]
        )

        # NOTE!!
        # IMPORTANT!!! set kernel diagonal to zero
        params["params"]["w"]["kernel"] = params["params"]["w"]["kernel"] - np.diag(
            np.diag(params["params"]["w"]["kernel"])
        )

        params["params"]["w"]["bias"] = (
            params["params"]["w"]["bias"] + lr * dparams["params"]["w"]["bias"]
        )

        return params

    def update_params(self, params, x, z, lr: float = 0.1):

        # input x of shape (batch, time, *features)
        # NOTE it has to be the one returned by the __call__ method tobe time-aligned with z

        # !!! so we don't even reshape it here !!!

        # z of shape (batch, time, features)

        dparams = self.compute_dparams(params, x, z)

        params = self.apply_dparams(params, dparams, lr)

        return params
