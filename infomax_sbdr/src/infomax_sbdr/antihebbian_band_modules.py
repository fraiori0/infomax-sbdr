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
    init_variance_w: float = 1.0
    scale_negative: float = 0.1
    weight_l2: float = 0.01

    def setup(self):
        self.wb = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(0.0),
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
        kernel = kernel / np.linalg.norm(kernel, axis=0, keepdims=True)

        # Center each band on the mean of projected data
        y = self.wb(x).reshape(-1, self.n_features)
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
        z = z.reshape(-1, z.shape[-1])

        # matrix of co-activations:
        zi_zj = z[..., :, None] * z[..., None, :]

        # # covariance matrix, instead
        # zi_zj = np.cov(z.T)

        mu_target = zi_zj.mean(axis=0)

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes

        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - momentum) * (mu_target - params["params"]["mu"])

        return dmu

    def compute_dwb(self, params, outs):
        # outs expected to have the same content/shape as returned from the __call__ method of this class
        x = outs["x"].reshape(-1, outs["x"].shape[-1])
        y = outs["y"].reshape(-1, outs["y"].shape[-1])
        z = outs["z"].reshape(-1, outs["z"].shape[-1])

        kernel = params["params"]["wb"]["kernel"]
        bias = params["params"]["wb"]["bias"]
        mu = params["params"]["mu"]
        mu_self = np.diag(params["params"]["mu"])

        # # # Compute parameter updates
        wTw = (kernel*kernel).sum(axis=-2)
        alpha = y / (wTw + 1e-4)
        # # Weights
        dw = - alpha[..., None, :] * (x[..., :, None] - alpha[..., None, :]*kernel)
        # Pull/push to co-activating samples depending on the error on co-activation statistics
        # only if units co-activated
        zi_zj = z[..., :, None] * z[..., None, :]
        p_err_ij = self.p_target**2 - mu
        dw_cross = dw[..., None] * (zi_zj * p_err_ij)[..., None, :, :]
        dw_cross = dw_cross.mean(axis=-1)
        # Also, always pull to activating samples
        dw_self = dw * z[..., None, :]
        # take mean over batches and add regularization term
        dw = 0.5*dw.mean(axis=0) + dw_self.mean(axis=0) - self.weight_l2 * kernel
        # # Bias
        # Pull/push to samples activating this unit depending on the error on activation statistics
        p_err_i = self.p_target - mu_self 
        db = - alpha * z # * p_err_i <--- !!! NOTE
        db = db.mean(axis=0) - self.weight_l2 * bias

        dwb = {
            "kernel": dw,
            "bias": db,
        }

        return dwb

    def compute_db2(self, params, outs):
        mu_self = np.diag(params["params"]["mu"])

        db2 = self.p_target - mu_self

        return db2

    def compute_dparams(self, params, outs, momentum=None):
        # x and z as returned from the __call__ method
        dparams = {
            "params": {
                "mu": None,
                "wb": {
                    "kernel": None,
                    "bias": None,
                },
                "b2": None,
            }
        }

        tmp_params = params.copy()

        if momentum is None:
            dmu = self.compute_dmu(params, outs, self.momentum)
        else:
            dmu = self.compute_dmu(params, outs, momentum)
        tmp_params["params"]["mu"] = params["params"]["mu"] + dmu

        # we do this because it should be preferable to first update mu,
        # and then use it to update the forward and lateral weights

        dwb = self.compute_dwb(tmp_params, outs)
        db2 = self.compute_db2(tmp_params, outs)

        dparams["params"]["mu"] = dmu
        dparams["params"]["wb"]["kernel"] = dwb["kernel"]
        dparams["params"]["wb"]["bias"] = dwb["bias"]
        dparams["params"]["b2"] = db2

        return dparams

    def apply_dparams(self, params, dparams, lr: float = 0.1):
        # update mu (exponential moving average of feature co-activations)
        params["params"]["mu"] = params["params"]["mu"] + dparams["params"]["mu"]

        # update hyperplane parameters
        params["params"]["wb"]["kernel"] = (
            params["params"]["wb"]["kernel"] + lr * dparams["params"]["wb"]["kernel"]
        )
        params["params"]["wb"]["bias"] = (
            params["params"]["wb"]["bias"] + lr * dparams["params"]["wb"]["bias"]
        )

        # update b2 (distance threshold)
        params["params"]["b2"] = (
            params["params"]["b2"] + 2 * lr * dparams["params"]["b2"]
        )

        # NOTE!! Clip b2 to be >= 0
        params["params"]["b2"] = np.clip(params["params"]["b2"], 0, None)

        return params

    def update_params(self, params, outs, lr: float = 0.1, momentum=None):

        dparams = self.compute_dparams(params, outs, momentum)

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
        if self.use_dropout:
            y = self.wb(self.dropout(x))
        else:
            y = self.wb(x)
        
        # divide by the squared norm of the weight vector of each unit
        # to get the distance from a plane
        kernel = self.variables["params"]["wb"]["kernel"]
        wTw = (kernel*kernel).sum(axis=-2)
        d_signed = y / wTw

        z = np.abs(d_signed) <=  self.b2

        return {
            "x": x, 
            "y": y,
            "z": z.astype(x.dtype),
            "d_signed": d_signed,
        }
    

