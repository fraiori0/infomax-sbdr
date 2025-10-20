import jax
import jax.numpy as np
import flax.linen as nn
from typing import Dict, Any


class AntiHebbianTDBase(nn.Module):
    """
    Base class for a Recurrent Centroid-based layer with Temporal Difference learning.
    
    This architecture combines:
    - Forward weights (W_f) as centroids with spherical receptive fields
    - Prediction weights (W_p) trained with TD learning to predict future activations
    
    Units activate when both conditions are met (AND operation):
    1. Input is within distance threshold of the centroid (forward activation)
    2. Activation is predicted based on discounted history (TD prediction)
    """
    
    n_features: int
    p_target: float
    gamma: float = 0.9
    momentum: float = 0.95
    n_input_features: int = None
    init_variance_w_forward: float = 1.0
    init_variance_w_prediction: float = 1.0
    
    def setup(self):
        # weights learning a TD prediction of the (binary) input
        self.w_td = nn.Dense(
            features=self.n_input_features,
            kernel_init=nn.initializers.variance_scaling(
                scale=self.init_variance_w_forward,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            use_bias=False
        )
        # forward weights
        self.w_f = nn.Dense(
            features=self.n_features,
            kernel_init=nn.initializers.variance_scaling( # AAAA
                scale=self.init_variance_w_forward,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            bias_init=nn.initializers.constant(0.0),
        )
        # lateral weights
        self.w_l = nn.Dense(
            features=self.n_features,
            # kernel_init=nn.initializers.variance_scaling(
            #     scale=self.init_variance_w_lateral,
            #     mode="fan_in",
            #     distribution="truncated_normal",
            # ),
            kernel_init=nn.initializers.constant(0.0),
            use_bias=False,
            # bias_init=nn.initializers.constant(self.bias_init_value),
        )
        
        
        # exponential-weighted moving average that tracks (co)activation of units
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
    
    
    def compute_dmu(self, params, outs, momentum):
        """
        Update running average of unit co-activations.
        
        Args:
            params: Current parameters
            z: Unit activations (*batch_dims, n_features)
            momentum: Momentum for exponential moving average
        
        Returns:
            dmu: Update to running average (n_features, n_features)
        """
        # Take the output activations
        z = outs["z"]
        # Matrix of co-activations
        zi_zj = z[..., :, None] * z[..., None, :]
        
        # Exponential moving average update
        dmu = (1.0 - momentum) * (zi_zj - params["params"]["mu"])
        
        # Average over all batch dimensions
        dmu = dmu.reshape(-1, self.n_features, self.n_features).mean(axis=0)
        
        return dmu
    
    def compute_dw_f(self, params, outs):
        """
        Compute updates for forward weights.

        Args:
            params: Current parameters
            outs: A dictionary, as returned by a forward pass or forward scan
        
        Returns:
            dw_f: Dictionary with kernel and bias updates
        """
        kernel = params["params"]["w_f"]["kernel"]  # (input_features, n_features)
        mu = params["params"]["mu"]  # (n_features, n_features)
        mu_self = np.diag(mu)  # (n_features,)

        z = outs["z"]  # (*batch, n_features)
        x = outs["x"]  # (*batch, input_features)
        
        # pulls kernel toward input if unit is active
        xi_zj = (x[..., :, None] - kernel) * z[..., None, :]
        # proportionally to the difference between the target and the actual co-activation
        dkernel = (self.p_target**2 - mu_self**2) * xi_zj
        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        # adjust bias to get an overall activity of p_target
        dbias = self.p_target - mu_self
        
        return {
            "kernel": dkernel,
            "bias": dbias,
        }
    
    def compute_dw_td(self, params, outs):
        """
        Compute updates for prediction weights using TD learning.
        
        Args:
            params: Current parameters
            outs: A dictionary, as returned by a forward pass or forward scan
        
        Returns:
            dw_td: Dictionary with kernel and bias updates
        """
        kernel = params["params"]["w_td"]["kernel"]  # (n_features, n_features)

        u_prev = outs["u_prev"]  # (*batch, n_features)
        td_err_prev = outs["td_err_prev"]  # (*batch, n_features)
        
        # Standard TD update: delta * u
        # u: (*batch, n_features), td_pred_err: (*batch, n_features)
        # We want: u[..., :, None] * td_pred_err[..., None, :]
        # Result: (*batch, n_features, n_features)
        ui_deltaj = u_prev[..., :, None] * td_err_prev[..., None, :]
        
        # Average over batch dimensions
        dkernel = ui_deltaj.reshape(-1, *kernel.shape).mean(axis=0)
        
        return {
            "kernel": dkernel,
            # NOTE: there is no bias in the prediction weights
        }
    
    def compute_dw_l(self, params, outs):
        """
        Compute updates for lateral weights.
        
        Args:
            params: Current parameters
            outs: A dictionary, as returned by a forward pass or forward scan
        
        Returns:
            dw_l: Dictionary with kernel and bias updates
        """

        mu = params["params"]["mu"]
        # mu_self = np.diag(params["params"]["mu"])

        kernel = params["params"]["w_l"]["kernel"]

        z = outs["z"]

        # matrix of co-active units, used to mask the updates
        # i.e., we update only weights between co-actived units
        zi_zj = z[..., :, None] * z[..., None, :]
        # proportionally to the difference between the target and the actual co-activation
        dkernel = (self.p_target**2 - mu) * zi_zj
        # mean on all batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        # adjust bias to get an overall activity of p_target
        # dbias = self.p_target - mu_self

        return {
            "kernel": dkernel,
        }
    
    def compute_dparams(self, params, outs, momentum=None):
        """
        Compute all parameter updates.
        
        Args:
            params: Current parameters
            outs: A dictionary, as returned by a forward pass or forward scan
            momentum: Optional momentum override
        
        Returns:
            dparams: Dictionary with all parameter updates
        """
        if momentum is None:
            momentum = self.momentum
        
        # Initialize update dictionary
        dparams = {
            "params": {
                "mu": None,
                "w_f": {
                    "kernel": None,
                    "bias": None,
                },
                "w_l": {
                    "kernel": None,
                },
                "w_td": {
                    "kernel": None,
                }
            }
        }
        
        # Create temporary params with updated mu
        # (Update mu first, then use it for weight updates)
        tmp_params = {
            "params": {
                key: val.copy() if isinstance(val, np.ndarray) else val
                for key, val in params["params"].items()
            }
        }
        dmu = self.compute_dmu(params, outs, momentum)
        tmp_params["params"]["mu"] = params["params"]["mu"] + dmu
        
        # Compute weight updates using updated mu
        dw_f = self.compute_dw_f(
            tmp_params, 
            outs,
        )
        dw_l = self.compute_dw_l(
            tmp_params,
            outs,
        )
        dw_td = self.compute_dw_td(
            params,
            outs
        )
        
        # Store all updates
        dparams["params"]["mu"] = dmu
        dparams["params"]["w_f"]["kernel"] = dw_f["kernel"]
        dparams["params"]["w_f"]["bias"] = dw_f["bias"]
        dparams["params"]["w_l"]["kernel"] = dw_l["kernel"]
        dparams["params"]["w_td"]["kernel"] = dw_td["kernel"]

        return dparams
    
    def apply_dparams(self, params, dparams, lr: float = 0.1):
        """
        Apply parameter updates.
        
        Args:
            params: Current parameters
            dparams: Parameter updates
            lr: Learning rate
        
        Returns:
            params: Updated parameters
        """
        # Update running average (no learning rate)
        params["params"]["mu"] = params["params"]["mu"] + dparams["params"]["mu"]
        
        # Update forward weights
        params["params"]["w_f"]["kernel"] = (
            params["params"]["w_f"]["kernel"] + lr * dparams["params"]["w_f"]["kernel"]
        )
        params["params"]["w_f"]["bias"] = (
            params["params"]["w_f"]["bias"] + lr * dparams["params"]["w_f"]["bias"]
        )
        
        # Update lateral weights
        params["params"]["w_l"]["kernel"] = (
            params["params"]["w_l"]["kernel"] + lr * dparams["params"]["w_l"]["kernel"]
        )

        # Update TD prediction weights
        params["params"]["w_td"]["kernel"] = (
            params["params"]["w_td"]["kernel"] + lr * dparams["params"]["w_td"]["kernel"]
        )

        # IMPORTANT!!! set kernel diagonal to zero for both lateral and prediction
        params["params"]["w_l"]["kernel"] = params["params"]["w_l"]["kernel"] - np.diag(
            np.diag(params["params"]["w_l"]["kernel"])
        )
        params["params"]["w_td"]["kernel"] = params["params"]["w_td"]["kernel"] - np.diag(
            np.diag(params["params"]["w_td"]["kernel"])
        )
        
        return params
    
    def update_params(self, params, outs, lr: float = 0.1, momentum=None):
        """
        Compute and apply parameter updates.
        
        Args:
            params: Current parameters
            outs: A dictionary, as returned by a forward pass or forward scan
            lr: Learning rate
            momentum: Optional momentum override
        
        Returns:
            params: Updated parameters
            dparams: Parameter updates (for debugging/logging)
        """
        dparams = self.compute_dparams(params, outs, momentum)
        params = self.apply_dparams(params, dparams, lr)
        
        return params, dparams


class AntiHebbianTDModule(AntiHebbianTDBase):
    """
    Recurrent module combining forward activation based on TD prediction and lateral activation.
    """
    
    def __call__(self, x, u_prev):
        """
        Single time step forward pass.
        
        Args:
            params: Model parameters
            x: Input (*batch_dims, input_features)
            u_prev: Previous discounted sum of activations (*batch_dims, n_features)
            # v_prev: Previous value prediction (*batch_dims, n_features)
            # y_p_prev: Prediction of activation, made from the previous step (*batch_dims, n_features) using v_prevprev and v_prev
        
        Returns:
            Dictionary with all activations and values
        """


        # Apply forward weights
        y = (self.w_f(x) > 0).astype(x.dtype)
        z = ((self.w_f(x) + self.w_l(y)) > 0).astype(x.dtype)

        # Update discounted sum of activations
        u = self.gamma * u_prev + x

        
        # Compute prediction from discounted trace of past activations
        v_prev = self.w_td(u_prev)
        # Compute value prediction
        v = self.w_td(u)
        # Compute the TD error on this time-step, ideally v_prev = z + self.gamma * v
        td_err_prev = (x + self.gamma * v) - v_prev
        
        return {
            # input/output for current time step
            "x": x,
            "y": y.astype(x.dtype),
            "z": z.astype(x.dtype),
            # discounted sum to be used at next time step
            "u": u,
            # values to be used for TD update
            "u_prev": u_prev,
            "td_err_prev": td_err_prev,
        }
    
    def forward_scan(self, x_seq, u_prev, key=None):
        """
        Apply network iteratively over a sequence using jax.lax.scan.
        
        Args:
            x_seq: Input sequence (*batch_dims, time_steps, input_features)
            u0: Initial discounted trace (*batch_dims, n_features)
            key: Random key (not used, but kept for interface compatibility)
        
        Returns:
            Dictionary with all sequences of activations
        """
        def f_scan(carry, input):
            x = input
            u_prev = carry
            out = self(x, u_prev)
            return out["u"], out
        
        # Move time axis to first position for scan
        x_seq = np.moveaxis(x_seq, -2, 0)
        
        # Perform scan
        _, outputs_seq = jax.lax.scan(
            f_scan, 
            u_prev, 
            x_seq
        )
        
        # Move time axis back to second-to-last position
        outputs_seq = jax.tree_util.tree_map(
            lambda arr: np.moveaxis(arr, 0, -2), 
            outputs_seq
        )
        
        return outputs_seq
    
    def generate_initial_state(self, key, x):
        """
        Generate initial states for u and v.
        
        Args:
            key: JAX random key
            x: Sample input to determine batch shape (*batch_dims, input_features)
        
        Returns:
            u0: Initial discounted sum (*batch_dims, n_features)
        """
        batch_shape = x.shape[:-1]
        state_shape = (*batch_shape, self.n_features)
        
        # initialize u with the expected value for a discounted sum of bernoulli variables each 
        # with a probability of self.p_target
        u_prev = np.ones(x.shape) * self.p_target/(1.0 - self.gamma)
        
        return {"u_prev": u_prev}