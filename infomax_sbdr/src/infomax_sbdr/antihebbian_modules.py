import jax
import jax.numpy as np
import flax.linen as nn
from functools import partial
from infomax_sbdr.utils import conv1d


""" -------------------- """
""" "Standard" Anti-Hebbian Modules" """
""" -------------------- """


class AntiHebbianBase(nn.Module):
    """Base class for a Anti-Hebbian Layer, with forward (input) and lateral weights."""

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

    def compute_dw_f(self, params, x, y, z):
        # x and z as returned from the __call__ method

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


class AntiHebbianModule(AntiHebbianBase):
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
    


class AntiHebbianTimeConvModule(AntiHebbianBase):
    """This layer also assume to have a second-to-last dimension corresponding to time.
    It will convolve over time on the inputs, computing a backward discounted sum.
    """

    gamma: float = 0.9
    seq_length: int = 20
    conv_mode: str = "valid"

    def setup(self):
        # setup the base class
        super().setup()

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
        # i.e., already flattened on the feature dimensions, if multiple (like image C, H, W)

        x = self.conv_op(x)
        y = self.w_f(x) > 0
        z = self.w_l(y) > 0
        z = np.logical_and(y, z)

        return x, y.astype(x.dtype), z.astype(x.dtype)



""" -------------------- """
""" Clone-Structured Anti-Hebbian Modules" """
""" -------------------- """


class AntiHebbianCloneStructuredBase(nn.Module):
    """Base class for a Clone-Structured Anti-Hebbian Layer, with forward (input) and lateral weights.

    The structure of this architecture assumes a total of n_clones*n_features hidden units.
        Hidden units are organized in "rows", with all units in a row sharing the same forward (input) weights.
        Lateral weights are individual for each hidden unit.

    The activation of a row is considered as a OR between all units in the row.
        As such, if each hidden units should have a probability p_target of activation,
        the row should have a probability of activation of 1-(1-p_target)**n_clones.

    Note, this is a recurrent architecture, as the previous activation of hidden units is used to compute the current activation through the lateral weights.
    The input also affects activation through the forward weights, in the same way for all hidden units in the same row.

    """

    n_features: int
    n_clones: int
    p_target: float
    p_target_row: float
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
            features=self.n_clones * self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w_lateral,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(self.bias_init_value),
        )

        # compute the target activation probabilty for a row
        assert (
            abs(self.p_target_row - (1 - ((1 - self.p_target) ** self.n_clones))) < 1e-2
        ), "The target activation probability for a row should be equal to 1-(1-p_target)**n_clones (tolerance of 1e-2)"

        # # store both probabilities inside the parameters of the module
        # self.p_target = self.param(
        #     "p_target", nn.initializers.constant(self.p_target), (), np.float32
        # )
        # self.p_target_row = self.param(
        #     "p_target_row", nn.initializers.constant(p_target_row), (), np.float32
        # )

        # Average Co-Activation Matrix for each individual unit
        mu0 = np.ones(
            (self.n_clones * self.n_features, self.n_clones * self.n_features)
        ) * self.p_target**2 + np.eye(self.n_clones * self.n_features) * (
            self.p_target - self.p_target**2
        )
        self.mu = self.param(
            "mu",
            # nn.initializers.constant(self.p_target),
            nn.initializers.constant(mu0),
            (self.n_clones * self.n_features, self.n_clones * self.n_features),
            np.float32,
        )

        # Average activity of each row
        self.mu_self_row = self.param(
            "mu_self_row",
            # nn.initializers.constant(self.p_target),
            nn.initializers.constant(self.p_target_row),
            (self.n_features,),
            np.float32,
        )

    def __call__(self, x, z_prev):
        raise NotImplementedError

    @nn.nowrap
    def gen_hidden_state(self, batch_shape=[]):
        return np.ones(
            (
                *batch_shape,
                self.n_clones * self.n_features,
            )
        )

    def flatten_z(self, z):
        return z.reshape(*z.shape[:-2], self.n_clones * self.n_features)

    def unflatten_z(self, z):
        return z.reshape(*z.shape[:-1], self.n_clones, self.n_features)

    def compute_z_row(self, z):
        # z as returned from the __call__ method
        # unflatten z
        z = self.unflatten_z(z)
        # project z using the OR on the "clone" axis
        z_row = 1.0 - np.prod(1.0 - z, axis=-2)
        return z_row

    def compute_dmu(self, params, z, momentum):
        # z as returned from the __call__ method

        # matrix of coactivations
        zi_zj = z[..., :, None] * z[..., None, :]

        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes

        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - momentum) * (zi_zj - params["params"]["mu"])
        # mean on all batch dimensions
        dmu = dmu.reshape(-1, z.shape[-1], z.shape[-1]).mean(axis=0)

        return dmu

    def compute_dmu_self_row(self, params, z_row, momentum):
        # z_row as computed from z as returned from the __call__ method

        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu_self_row = (1.0 - momentum) * (z_row - params["params"]["mu_self_row"])
        # mean on all batch dimensions
        dmu_self_row = dmu_self_row.reshape(-1, z_row.shape[-1]).mean(axis=0)

        return dmu_self_row

    def compute_dw_f(self, params, x, y, z, z_row):
        # x as returned from the __call__ method
        # z_row as computed from z as returned from the __call__ method

        kernel = params["params"]["w_f"]["kernel"]
        mu_self = params["params"]["mu_self_row"]

        xi_zj = (x[..., :, None] - kernel) * z_row[..., None, :]

        dkernel = (self.p_target_row**2 - mu_self**2) * xi_zj
        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        dbias = self.p_target_row - mu_self

        dw_f = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw_f

    def compute_dw_l(self, params, x, y, z, z_row):
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
        # x, y, and z as returned from the __call__ method

        # compute z_row
        z_row = self.compute_z_row(z)

        dparams = {
            "params": {
                "mu": None,
                "mu_self_row": None,
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

        # First update the runnig averages, than we use it to update the
        # forward and lateral weights

        if momentum is None:
            dmu = self.compute_dmu(params, z, self.momentum)
            dmu_self_row = self.compute_dmu_self_row(params, z_row, self.momentum)
        else:
            dmu = self.compute_dmu(params, z, momentum)
            dmu_self_row = self.compute_dmu_self_row(params, z_row, momentum)

        tmp_params["params"]["mu"] = params["params"]["mu"] + dmu
        tmp_params["params"]["mu_self_row"] = (
            params["params"]["mu_self_row"] + dmu_self_row
        )

        dw_f = self.compute_dw_f(tmp_params, x, y, z, z_row)
        dw_l = self.compute_dw_l(tmp_params, x, y, z, z_row)

        dparams["params"]["mu"] = dmu
        dparams["params"]["mu_self_row"] = dmu_self_row
        dparams["params"]["w_f"]["kernel"] = dw_f["kernel"]
        dparams["params"]["w_f"]["bias"] = dw_f["bias"]
        dparams["params"]["w_l"]["kernel"] = dw_l["kernel"]
        dparams["params"]["w_l"]["bias"] = dw_l["bias"]

        return dparams

    def apply_dparams(self, params, dparams, lr: float = 0.1):
        # update mu (exponential moving average of hidden units co-activations)
        params["params"]["mu"] = params["params"]["mu"] + dparams["params"]["mu"]

        # update mu_self_row (exponential moving average of row activations)
        params["params"]["mu_self_row"] = (
            params["params"]["mu_self_row"] + dparams["params"]["mu_self_row"]
        )

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
        # IMPORTANT!!! set the lateral weights kernel diagonal to zero, so units do not self-activate
        params["params"]["w_l"]["kernel"] = params["params"]["w_l"]["kernel"] - np.diag(
            np.diag(params["params"]["w_l"]["kernel"])
        )

        return params

    def update_params(self, params, x, y, z, lr: float = 0.1, momentum=None):

        dparams = self.compute_dparams(params, x, y, z, momentum)

        params = self.apply_dparams(params, dparams, lr)

        return params, dparams

    def sequence_scan(self, x_seq, z0):
        """
        Iterative forward pass of the model scanning over a sequence.

        Args:
        - x_seq: np.ndarray (*batch_sizes, seq_length, input_size)
            Input sequence
        - z0: np.ndarray (*batch_sizes, self.n_clones*self.n_features)
            Initial value of the hidden units

        Returns:
        - x_seq: np.ndarray (*batch_sizes, seq_length, input_size)
            Input sequence
        - y_seq: np.ndarray (*batch_sizes, seq_length, self.n_features)
            Forward activation sequence
        - z_seq: np.ndarray (*batch_sizes, seq_length, self.n_clones*self.n_features)
            Hidden units activation sequence
        """

        def f_scan(carry, inputs):
            x = inputs
            z_prev = carry
            x, y, z = self(x, z_prev)
            return z, (y, z)

        # NOTE: we place time axis as first, as jax.lax.scan requires to scan over the first axis of an array
        x_seq = np.moveaxis(x_seq, -2, 0)
        z0
        _, (y_seq, z_seq) = jax.lax.scan(f_scan, z0, x_seq)

        # put back time axis
        x_seq = np.moveaxis(x_seq, 0, -2)
        y_seq = np.moveaxis(y_seq, 0, -2)
        z_seq = np.moveaxis(z_seq, 0, -2)

        return x_seq, y_seq, z_seq


class AntiHebbianCloneStructuredModule(AntiHebbianCloneStructuredBase):

    def __call__(self, x, z_prev):

        y = self.w_f(x) > 0
        z = self.w_l(z_prev) > 0

        z = np.logical_and(y[..., None, :], self.unflatten_z(z))

        z = self.flatten_z(z)

        # # if some of the array of activation in the batch are all zeros, set such array to all 1s
        # mask_notallzeros = np.any(z, axis=-1, keepdims=True)
        # z = np.logical_or(
        #     np.logical_and(z, mask_notallzeros),
        #     np.ones(z.shape, dtype=bool) * np.logical_not(mask_notallzeros),
        # )

        return x, y.astype(x.dtype), z.astype(x.dtype)


""" -------------------- """
""" Clone-Structured Anti-Hebbian Modules" """
""" -------------------- """


class AntiHebbianCloneStructuredTemporalBase(nn.Module):
    """
    Base class for a modified version of the Clone-Structured Anti-Hebbian Layer.

    Her we consider each "row" as having both the forward and lateral weights,
    along with an additional set of temporal weights separate for each clone, that should learn to make prediction in the "clone" space.

    Note, this is a recurrent architecture, as the previous activation of hidden units is used to compute the prediction through the temporal weights.

    """

    n_features: int
    n_clones: int
    p_target: float
    momentum: float = 0.95
    init_variance_w_forward: float = 1.0
    init_variance_w_lateral: float = 1.0
    init_variance_w_temporal: float = 1.0
    bias_init_value: float = 0.0

    def setup(self):
        # foward
        self.w_f = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w_forward,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(self.bias_init_value),
        )
        # lateral
        self.w_l = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w_lateral,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(self.bias_init_value),
        )
        # temporal
        self.w_t = nn.Dense(
            features=self.n_clones * self.n_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w_temporal,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(self.bias_init_value),
        )

        # Average Co-Activation Matrix for each individual row
        mu0_inter_row = np.ones(
            (self.n_features, self.n_features)
        ) * self.p_target**2 + np.eye(self.n_features) * (
            self.p_target - self.p_target**2
        )
        self.mu_inter_row = self.param(
            "mu_inter_row",
            nn.initializers.constant(mu0_inter_row),
            (self.n_features, self.n_features),
            np.float32,
        )

        # Avergae error for each individual unit
        self.mu_error = self.param(
            "mu_error",
            nn.initializers.constant(0.0),
            (self.n_features * self.n_clones),
            np.float32,
        )

    def __call__(self, x, z_prev):

        # actual activation
        y_row = self.w_f(x) > 0
        z_row = self.w_l(y_row) > 0
        z_row = np.logical_and(y_row, z_row)

        # prediction
        z_pred = self.w_t(z_prev) > 0
        z_row_pred = np.any(self.unflatten_z(z_pred), axis=-1)

        # Actual activation
        z = np.logical_or(
            # Active units in a row correctly predicted as active
            self.flatten_z(
                np.logical_and(
                    np.logical_and(z_row, z_row_pred)[..., None],
                    self.unflatten_z(z_pred),
                )
            ),
            # All units in a row wrongly predicted as inactive
            self.flatten_z(
                np.logical_and(
                    np.logical_and(z_row, np.logical_not(z_row_pred))[..., None],
                    np.logical_not(self.unflatten_z(z_pred)),
                )
            ),
        )

        return x, z_prev, z.astype(x.dtype), z_pred.astype(x.dtype)

    @nn.nowrap
    def gen_hidden_state(self, batch_shape=[]):
        return np.zeros((*batch_shape, self.n_features * self.n_clones))

    @nn.nowrap
    def flatten_z(self, z):
        return z.reshape(*z.shape[:-2], self.n_features * self.n_clones)

    @nn.nowrap
    def unflatten_z(self, z):
        return z.reshape(*z.shape[:-1], self.n_features, self.n_clones)

    @nn.nowrap
    def compute_z_row(self, z):
        # z as returned from the __call__ method
        # unflatten z
        z = self.unflatten_z(z)
        # project z using the OR on the "clone" axis
        # z_row = 1.0 - np.prod(1.0 - z, axis=-1)
        z_row = np.any(z > 1e-3, axis=-1)
        return z_row.astype(z.dtype)

    def compute_dmu_inter_row(self, params, z_row, momentum):
        # # z_row as computed from z as returned from the __call__ method
        # matrix of coactivations
        zi_zj = z_row[..., :, None] * z_row[..., None, :]
        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes
        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - momentum) * (zi_zj - params["params"]["mu_inter_row"])
        # mean on all batch dimensions
        dmu = dmu.reshape(-1, self.n_features, self.n_features).mean(axis=0)
        return dmu

    def compute_dmu_error(self, params, z_error, momentum):
        # mean recursive estimator
        # https://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes
        # NOTE: we exploit automatic broadcasting here, which follows NumPy rules
        dmu = (1.0 - momentum) * (z_error - params["params"]["mu_error"])
        # mean on all batch dimensions
        dmu = dmu.reshape((-1, self.n_features * self.n_clones)).mean(axis=0)
        return dmu

    def compute_dw_f(self, params, x, z_row):

        kernel = params["params"]["w_f"]["kernel"]
        mu_self = np.diag(params["params"]["mu_inter_row"])

        xi_zj = (x[..., :, None] - kernel) * z_row[..., None, :]

        dkernel = (self.p_target**2 - mu_self**2) * xi_zj
        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        dbias = self.p_target - mu_self

        dw_f = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw_f

    def compute_dw_l(self, params, z_row):

        mu = params["params"]["mu_inter_row"]
        mu_self = np.diag(params["params"]["mu_inter_row"])

        kernel = params["params"]["w_l"]["kernel"]

        # matrix of co-active units, used to mask the updates
        # i.e., we update only weights between co-actived units
        zi_zj = z_row[..., :, None] * z_row[..., None, :]

        dkernel = (self.p_target**2 - mu) * zi_zj

        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        dbias = self.p_target - mu_self

        dw_l = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw_l

    def compute_dw_t(self, params, z, z_prev):
        kernel = params["params"]["w_t"]["kernel"]

        # z_error = z - z_pred

        # note, kernel is of shape (input_features, output_features)
        # z_error gives the sign of the update to each output features
        # z_prev gives the target of the update
        zi_zj = (z_prev[..., :, None] - kernel) * (z[..., None, :])
        # scale
        dkernel = zi_zj * self.p_target / self.n_clones
        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)

        # update the bias in the direction opposite to the running average of the error of this unit
        dbias = params["params"]["mu_error"]

        dw_t = {
            "kernel": dkernel,
            "bias": dbias,
        }

        return dw_t

    def compute_dparams(self, params, output, momentum=None):

        dparams = {
            "params": {
                "mu": None,
                "mu_self_row": None,
                "w_f": {
                    "kernel": None,
                    "bias": None,
                },
                "w_l": {
                    "kernel": None,
                    "bias": None,
                },
                "w_t": {
                    "kernel": None,
                    "bias": None,
                },
            }
        }

        tmp_params = params.copy()

        x, z_prev, z, z_pred = output
        z_row = self.compute_z_row(z)
        z_error = z - z_pred

        # First update the runnig averages, then we use them to update the weights

        if momentum is None:
            dmu_inter_row = self.compute_dmu_inter_row(params, z_row, self.momentum)
            dmu_error = self.compute_dmu_error(params, z_error, 0.9)  # self.momentum)
        else:
            dmu_inter_row = self.compute_dmu_inter_row(params, z_row, momentum)
            dmu_error = self.compute_dmu_error(params, z_error, 0.9)  # momentum)

        tmp_params["params"]["mu_inter_row"] = (
            params["params"]["mu_inter_row"] + dmu_inter_row
        )
        tmp_params["params"]["mu_error"] = params["params"]["mu_error"] + dmu_error

        dw_f = self.compute_dw_f(tmp_params, x, z_row)
        dw_l = self.compute_dw_l(tmp_params, z_row)
        dw_t = self.compute_dw_t(tmp_params, z, z_prev)

        dparams["params"]["mu_inter_row"] = dmu_inter_row
        dparams["params"]["mu_error"] = dmu_error
        dparams["params"]["w_f"]["kernel"] = dw_f["kernel"]
        dparams["params"]["w_f"]["bias"] = dw_f["bias"]
        dparams["params"]["w_l"]["kernel"] = dw_l["kernel"]
        dparams["params"]["w_l"]["bias"] = dw_l["bias"]
        dparams["params"]["w_t"]["kernel"] = dw_t["kernel"]
        dparams["params"]["w_t"]["bias"] = dw_t["bias"]

        return dparams

    def apply_dparams(self, params, dparams, lr: float = 0.01):
        params["params"]["mu_inter_row"] = (
            params["params"]["mu_inter_row"] + dparams["params"]["mu_inter_row"]
        )

        params["params"]["mu_error"] = (
            params["params"]["mu_error"] + dparams["params"]["mu_error"]
        )

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
        # IMPORTANT!!! set the lateral weights kernel diagonal to zero, so units do not self-activate
        params["params"]["w_l"]["kernel"] = params["params"]["w_l"]["kernel"] - np.diag(
            np.diag(params["params"]["w_l"]["kernel"])
        )

        # update w_t (temporal prediction weights)
        params["params"]["w_t"]["kernel"] = (
            params["params"]["w_t"]["kernel"] + lr * dparams["params"]["w_t"]["kernel"]
        )
        params["params"]["w_t"]["bias"] = (
            params["params"]["w_t"]["bias"] + lr * dparams["params"]["w_t"]["bias"]
        )
        params["params"]["w_t"]["kernel"] = params["params"]["w_t"]["kernel"] - np.diag(
            np.diag(params["params"]["w_t"]["kernel"])
        )

        return params

    def update_params(self, params, output, lr: float = 0.01, momentum=None):

        dparams = self.compute_dparams(params, output, momentum)

        params = self.apply_dparams(params, dparams, lr)

        return params, dparams

    def sequence_scan(self, x_seq, z0):
        """
        Iterative forward pass of the model scanning over a sequence.

        Args:
        - x_seq: np.ndarray (*batch_sizes, seq_length, input_size)
            Input sequence
        - z0: np.ndarray (*batch_sizes, self.n_clones*self.n_features)
            Initial value of the hidden units

        Returns:
        - output_seq
        """

        def f_scan(carry, inputs):
            x = inputs
            z_prev = carry
            x, z_prev, z, z_pred = self(x, z_prev)
            return z, (x, z_prev, z, z_pred)

        # NOTE: we place time axis as first, as jax.lax.scan requires to scan over the first axis of an array
        x_seq = np.moveaxis(x_seq, -2, 0)
        _, (x_seq, z_prev_seq, z_seq, z_pred_seq) = jax.lax.scan(f_scan, z0, x_seq)

        # put back time axis
        x_seq = np.moveaxis(x_seq, 0, -2)
        z_prev_seq = np.moveaxis(z_prev_seq, 0, -2)
        z_seq = np.moveaxis(z_seq, 0, -2)
        z_pred_seq = np.moveaxis(z_pred_seq, 0, -2)

        return x_seq, z_prev_seq, z_seq, z_pred_seq


class AntiHebbianCloneStructuredTemporalModule(AntiHebbianCloneStructuredTemporalBase):
    pass
