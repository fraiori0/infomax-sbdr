import jax
import jax.numpy as np
from jax import jit, grad, vmap
import sparse_distributed_memory.binary_comparisons as bn
import flax.linen as nn
from typing import Sequence, Callable
from functools import partial
from flax.core import freeze, unfreeze, FrozenDict
from sparse_distributed_memory.utils import conv1d


class FoldiakLayer(nn.Module):
    """Layer that implements feedforward connection and lateral inhibition, with thresholding.
    Note, gradient descent will not work with this layer, as it is not differentiable.
    """

    n_features: int
    p_target: float
    momentum: float = 0.95
    init_variance_q: float = 1.0
    init_variance_w: float = 0.1
    lr_scale_w: float = 5.0

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
        # input x of shape (*batch, features)
        # i.e., already flattened on the feature dimensions, if multiple (like image C, H, W)

        y = self.q(x) > 0
        z = self.w(y) > 0
        z = y * z

        return z

    def compute_dmu(self, params, z):
        # input z of shape (*batch, features)

        # matrix of coactivations
        zi_zj = z[..., :, None] * z[..., None, :]

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes

        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - self.momentum) * (zi_zj - params["params"]["mu"])
        # mean on all batch dimensions
        dmu = dmu.reshape(-1, z.shape[-1], z.shape[-1]).mean(axis=0)

        return dmu

    def compute_dq(self, params, x, z):
        # input x of shape (*batch, features)
        # input z of shape (*batch, features)

        q_kernel = params["params"]["q"]["kernel"]
        q_bias = params["params"]["q"]["bias"]
        mu_self = np.diag(params["params"]["mu"])

        # with weight decay
        # (note: not weight, we can interpret it as pulling the weights towards the value of x, like an error term)
        # indeed, if the weights are equal to x, xi_zj = 0, and the weights are not updated
        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        xi_zj = (x[..., :, None] - q_kernel) * z[..., None, :]

        dkernel = (self.p_target**2 - mu_self**2) * xi_zj
        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *q_kernel.shape).mean(axis=0)
        dbias = self.p_target - mu_self

        dq = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dq

    def compute_dw(self, params, z):
        # input z of shape (*batch, features)

        mu = params["params"]["mu"]
        mu_self = np.diag(params["params"]["mu"])

        w_kernel = params["params"]["w"]["kernel"]

        zi_zj = z[..., :, None] * z[..., None, :]

        dkernel = (self.p_target**2 - mu) * zi_zj
        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *w_kernel.shape).mean(axis=0)
        dbias = self.p_target - mu_self

        dw = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw

    def compute_dparams(self, params, x, z):
        # input x of shape (*batch, features)
        # input z of shape (*batch, features)

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

    def update_params(self, params, x, z, lr: float = 0.1):
        # input x of shape (*batch, features)
        # input z of shape (*batch, features)

        dparams = self.compute_dparams(params, x, z)

        params = params

        # update mu (exponential moving average of feature co-activations)
        params["params"]["mu"] = params["params"]["mu"] + dparams["params"]["mu"]

        # update q (feedforward weights)
        params["params"]["q"]["kernel"] = (
            params["params"]["q"]["kernel"] + lr * dparams["params"]["q"]["kernel"]
        )
        params["params"]["q"]["bias"] = (
            params["params"]["q"]["bias"] + lr * dparams["params"]["q"]["bias"]
        )

        # update w (lateral weights)
        # NOTE, we scale the learning rate for w
        # the idea is that we don't need necesarrily sparse receptive field for q (input),
        # as long as the inhibition is effective in producing a sparse final activation.
        params["params"]["w"]["kernel"] = (
            params["params"]["w"]["kernel"]
            + self.lr_scale_w * lr * dparams["params"]["w"]["kernel"]
        )
        params["params"]["w"]["bias"] = (
            params["params"]["w"]["bias"]
            + self.lr_scale_w * lr * dparams["params"]["w"]["bias"]
        )
        # important, set kernel diagonal to zero
        params["params"]["w"]["kernel"] = params["params"]["w"]["kernel"] - np.diag(
            np.diag(params["params"]["w"]["kernel"])
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
    init_variance_w: float = 0.1
    gamma: float = 0.9
    seq_length: int = 40
    conv_mode: str = "valid"
    lr_scale_w: float = 5.0

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
        # input x of shape (*batch, time, features)

        # apply the convolution operator
        x = self.conv_op(x)

        # apply the feedforward weights
        y = self.q(x) > 0

        # TODO: should we convolve over time here as well?

        # apply the lateral connections
        z = self.w(y) > 0

        # activate only if both y and z are active
        z = y * z

        # return also the convolved x, which was the actual input to the units
        # we want to use that for the update
        return x, z

    def compute_dmu(self, params, z):
        # input z assumed to be of shape (*batch_dims, time, features)
        # with time we added two "None" dimensions when subtracting the current value of mu
        # and take the mean over the first two axis

        # matrix of coactivations
        zi_zj = z[..., :, None] * z[..., None, :]

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes
        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - self.momentum) * (zi_zj - params["params"]["mu"])
        # mean on all batch dimensions
        dmu = dmu.reshape(-1, z.shape[-1], z.shape[-1]).mean(axis=0)

        return dmu

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

        d_kernel = (self.p_target**2 - mu_self**2) * xi_zj
        d_kernel = d_kernel.reshape(-1, *q_kernel.shape).mean(axis=0)
        d_bias = self.p_target - mu_self

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
        # update mu (exponential moving average of feature co-activations)
        params["params"]["mu"] = params["params"]["mu"] + dparams["params"]["mu"]
        # update q (feedforward weights)
        params["params"]["q"]["kernel"] = (
            params["params"]["q"]["kernel"] + lr * dparams["params"]["q"]["kernel"]
        )
        params["params"]["q"]["bias"] = (
            params["params"]["q"]["bias"] + lr * dparams["params"]["q"]["bias"]
        )
        # update w (lateral weights)
        # NOTE, we scale the learning rate for w
        # the idea is that we don't need necesarrily sparse receptive field for q (input),
        # as long as the inhibition is effective in producing a sparse final activation.
        params["params"]["w"]["kernel"] = (
            params["params"]["w"]["kernel"]
            + self.lr_scale_w * lr * dparams["params"]["w"]["kernel"]
        )
        params["params"]["w"]["bias"] = (
            params["params"]["w"]["bias"]
            + self.lr_scale_w * lr * dparams["params"]["w"]["bias"]
        )

        # NOTE!!
        # IMPORTANT!!! set kernel diagonal to zero
        params["params"]["w"]["kernel"] = params["params"]["w"]["kernel"] - np.diag(
            np.diag(params["params"]["w"]["kernel"])
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


class FoldiakTest(nn.Module):
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
        # input x of shape (*batch, features)
        # i.e., already flattened on the feature dimensions, if multiple (like image C, H, W)

        y = self.q(x) > 0
        z = self.w(y) > 0
        z = y * z

        return z

    def compute_dmu(self, params, z):
        # input z of shape (*batch, features)

        # matrix of coactivations
        zi_zj = z[..., :, None] * z[..., None, :]

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes

        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - self.momentum) * (zi_zj - params["params"]["mu"])
        # mean on all batch dimensions
        dmu = dmu.reshape(-1, z.shape[-1], z.shape[-1]).mean(axis=0)

        return dmu

    def compute_dq(self, params, x, z):
        # input x of shape (*batch, x_features)
        # input z of shape (*batch, z_features)

        q_kernel = params["params"]["q"]["kernel"]
        q_bias = params["params"]["q"]["bias"]
        mu_self = np.diag(params["params"]["mu"])

        pz_tot = z.sum(axis=-1) / self.n_features

        # with weight decay
        # (note: not weight, we can interpret it as pulling the weights towards the value of x, like an error term)
        # indeed, if the weights are equal to x, xi_zj = 0, and the weights are not updated
        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        # active units will be affected by the unit activity
        xi_zj_unit = (x[..., :, None] - q_kernel) * z[
            ..., None, :
        ]  # (*batch, x_features, z_features)
        p_target_q = self.p_target + 1.0 / self.n_features
        unit_p_error = p_target_q**2 - mu_self**2
        dkernel_unit = unit_p_error * xi_zj_unit

        # # non-active units will be affected by the total activity
        # xi_zj_sample = (x[..., :, None] - q_kernel) * (
        #     1.0 - z[..., None, :]
        # )  # (*batch, x_features, z_features)
        # # proportionally to the difference between the target and the actual total activity expected for a sample
        # sample_p_error = np.abs(self.p_target - pz_tot)  # (*batch,)
        # dkernel_sample = (
        #     sample_p_error[..., None, None] * xi_zj_sample
        # )  # (*batch, x_features, z_features)
        # # sum the two contributions
        dkernel = dkernel_unit  # + self.p_target * dkernel_sample

        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *q_kernel.shape).mean(axis=0)
        dbias = p_target_q - mu_self

        dq = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dq

    def compute_dw(self, params, z):
        # input z of shape (*batch, features)

        mu = params["params"]["mu"]
        mu_self = np.diag(params["params"]["mu"])

        w_kernel = params["params"]["w"]["kernel"]

        zi_zj_unit = z[..., :, None] * z[..., None, :]  # (*batch, features, features)
        unit_pipj_error = self.p_target**2 - mu**2
        dkernel_unit = unit_pipj_error * zi_zj_unit

        # # non-co-active units will be affected by the total activity
        # pz_tot = z.sum(axis=-1) / self.n_features  # (*batch,)
        # zi_zj_sample = z[..., :, None] * (
        #     1.0 - z[..., None, :]
        # )  # (*batch, features, features)
        # sample_pipj_error = self.p_target**2 - pz_tot**2  # (*batch,)
        # dkernel_sample = sample_pipj_error[..., None, None] * (1.0 - zi_zj_sample)

        dkernel = dkernel_unit  # + 4 * self.p_target * dkernel_sample

        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *w_kernel.shape).mean(axis=0)
        dbias = self.p_target - mu_self

        dw = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw

    def compute_dparams(self, params, x, z):
        # input x of shape (*batch, features)
        # input z of shape (*batch, features)

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

    def update_params(self, params, x, z, lr: float = 0.1):
        # input x of shape (*batch, features)
        # input z of shape (*batch, features)

        dparams = self.compute_dparams(params, x, z)

        params = params

        params["params"]["mu"] = params["params"]["mu"] + dparams["params"]["mu"]
        params["params"]["q"]["kernel"] = (
            params["params"]["q"]["kernel"] + lr * dparams["params"]["q"]["kernel"]
        )
        params["params"]["q"]["bias"] = (
            params["params"]["q"]["bias"] + lr * dparams["params"]["q"]["bias"]
        )
        params["params"]["w"]["kernel"] = (
            params["params"]["w"]["kernel"] + lr * dparams["params"]["w"]["kernel"]
        )
        # important, set kernel diagonal to zero
        params["params"]["w"]["kernel"] = params["params"]["w"]["kernel"] - np.diag(
            np.diag(params["params"]["w"]["kernel"])
        )
        params["params"]["w"]["bias"] = (
            params["params"]["w"]["bias"] + lr * dparams["params"]["w"]["bias"]
        )

        return params
