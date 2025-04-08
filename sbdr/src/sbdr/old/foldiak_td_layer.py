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


class FoldiakTDLayer(nn.Module):
    """Layer that implements feedforward connection and lateral inhibition, with thresholding.
    Note, gradient descent will not work with this layer, as it is not differentiable.

    The TD FOldiak Layer will not track the mean activation, but the Successor Value from each feature to each feature,
    as would be done in a tabular case, using eligibility traces and a TD-lambda update rule.
    """

    n_features: int
    p_target: float
    momentum: float = 0.95
    init_variance_q: float = 1.0
    init_variance_w: float = 1.0
    gamma_td: float = 0.9
    lambda_td: float = 0.9

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

        # # Eligibility traces
        # self.e = self.param(
        #     "e",
        #     nn.initializers.constant(
        #         self.p_target/(1.0-self.gamma_td*self.lambda_td)),
        #     (self.n_features,),
        #     np.float32,
        # )

        # Successor values matrix
        self.mu = self.param(
            "mu",
            nn.initializers.constant(self.p_target/(1.0-self.gamma_td)),
            (self.n_features, self.n_features),
            np.float32,
        )

        # if units fire homogeneously, the mean activation should be p_target
        # and the discounted sum would be p_target/(1.0-self.gamma_td)
        self.p_discounted = self.p_target/(1.0-self.gamma_td)

    def __call__(self, x):
        # input x of shape (*batch, features)

        # apply the feedforward weights
        y = self.q(x) > 0

        # apply the lateral connections
        z = self.w(y) > 0

        # activate only if both y and z are active
        z = y * z

        return z

    def compute_de(self, e, z):
        # # Compute the update to be applied to the eligibility traces
        # input e assumed to be of shape (*batch_dims, features)
        # input z assumed to be of shape (*batch_dims, features)

        de = (self.gamma_td*self.lambda_td - 1.0) * e + z

        return de

    def compute_dmu(self, params, e, z_pre, z_post):
        # input z_pre assumed to be of shape (*batch_dims, features)
        # input z_post assumed to be of shape (*batch_dims, features)
        # input e assumed to be be of shape (*batch_dims, features)

        mu = params["params"]["mu"]

        # mean of the rows of mu weighted by z_pre. shape (*batch_dims, features)
        mean_mu_pre = (z_pre[..., :, None] *
                       mu).reshape(-1, *mu.shape).sum(axis=-2)
        mean_mu_pre = mean_mu_pre/(z_pre.sum(axis=-1)[..., None])
        # mean of the rows of mu weighted by z_pos. shape (*batch_dims, features)
        mean_mu_post = (z_post[..., :, None] *
                        mu).reshape(-1, *mu.shape).sum(axis=-2)
        mean_mu_post = mean_mu_post/(z_post.sum(axis=-1)[..., None])

        # shape (*batch_dims, features)
        td_error = z_post + self.gamma_td * mean_mu_post - mean_mu_pre

        # shape (*batch_dims, features, features)
        dmu = td_error[..., None, :] * e[..., :, None]

        # mean update due to all the samples in the batch
        return dmu.reshape(-1, *mu.shape).mean(axis=0)

    def compute_dq(self, params, x, z):
        # input x of shape (*batch_dims, time, *features)
        # NOTE it has to be the one returned by the __call__ method tobe time-aligned with z

        # !!! so we don't even reshape it here !!!

        # z of shape (*batch_dims, time, features)

        q_kernel = params["params"]["q"]["kernel"]
        q_bias = params["params"]["q"]["bias"]
        mu_self = np.diag(params["params"]["mu"])

        # with weight decay
        xi_zj = (x[..., :, None] - q_kernel) * z[..., None, :]

        d_kernel = (self.p_discounted**2 - mu_self**2) * xi_zj
        d_kernel = d_kernel.reshape(-1, *q_kernel.shape).mean(axis=0)
        d_bias = self.p_discounted - mu_self

        dq = {
            "kernel": d_kernel,
            "bias": d_bias,
        }

        return dq

    def compute_dw(self, params, z):
        # input z of shape (*batch_dims, time, features)

        mu = params["params"]["mu"]
        mu_self = np.diag(params["params"]["mu"])

        w_kernel = params["params"]["w"]["kernel"]

        zi_zj = z[..., :, None] * z[..., None, :]
        d_kernel = (self.p_target**2 - mu) * zi_zj
        d_kernel = d_kernel.reshape(-1, *w_kernel.shape).mean(axis=0)
        d_bias = self.p_target - mu_self

        dw = {
            "kernel": d_kernel,
            "bias": d_bias,
        }

        return dw

    def compute_dparams(self, params, x, z):
        # input x of shape (*batch, time, *features)
        # NOTE it has to be the one returned by the __call__ method tobe time-aligned with z

        # !!! so we don't even reshape it here !!!

        # z of shape (*batch, time, features)

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

        tmp_params = params.copy()

        dmu = self.compute_dmu(params, z)
        tmp_params["params"]["mu"] = params["params"]["mu"] + dmu

        # we do this because it should be preferable to update mu and then use it
        # to compute the update for q and w

        dq = self.compute_dq(tmp_params, x, z)
        dw = self.compute_dw(tmp_params, z)

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
