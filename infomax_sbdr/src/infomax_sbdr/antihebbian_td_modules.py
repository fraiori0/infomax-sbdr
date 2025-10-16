import jax
import jax.numpy as np
import flax.linen as nn
from typing import Dict, Any


class CentroidTDBase(nn.Module):
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
        # Forward weights as centroids - using Dense layer for consistency
        # The kernel will be transposed: Dense uses (input_features, output_features)
        # We'll interpret each column as a centroid in input space
        # Forward weights (centroids)
        self.kernel_f = self.param(
            'w_f_kernel',
            nn.initializers.variance_scaling(
                scale=self.init_variance_w_forward,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            (self.n_input_features, self.n_features),  # (input_features, n_features)
        )
        
        self.bias_f = self.param(
            'w_f_bias',
            nn.initializers.constant(0.5 * np.sqrt(self.n_input_features)),
            (self.n_features,),
        )
    
        # Prediction weights
        self.kernel_p = self.param(
            'w_p_kernel',
            nn.initializers.variance_scaling(
                scale=self.init_variance_w_prediction,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            (self.n_features, self.n_features),
        )
        
        self.bias_p = self.param(
            'w_p_bias',
            nn.initializers.constant(0.0),
            (self.n_features,),
        )
        # running exponential-weihgted mean that tracks (co)activation of units
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
        Compute updates for forward (centroid) weights.
        
        Combines two objectives:
        1. Self-activity: Move centroids toward/away from inputs based on activity level
        2. Decorrelation: Push/pull centroids apart based on co-activation statistics
        
        Args:
            params: Current parameters
            outs: A dictionary, as returned by a forward pass or forward scan
        
        Returns:
            dw_f: Dictionary with kernel and bias updates
        """
        kernel = params["params"]["w_f_kernel"]  # (input_features, n_features)
        mu = params["params"]["mu"]  # (n_features, n_features)
        mu_self = np.diag(mu)  # (n_features,)

        z = outs["z"]  # (*batch, n_features)
        x = outs["x"]  # (*batch, input_features)
        
        # Self-activity term: move centroids based on activity level
        # (x - kernel) gives direction from centroid to input
        # Shape: x is (*batch, input_features), kernel is (input_features, n_features)
        # We want: (x[..., :, None] - kernel) * z[..., None, :]
        # Result: (*batch, input_features, n_features)
        xi_minus_wfi = x[..., :, None] - kernel
        self_activity_scale = (self.p_target - mu_self)
        self_activity_term = self_activity_scale * xi_minus_wfi * z[..., None, :]
        
        # Decorrelation term: push/pull centroids apart based on co-activation
        # For each pair of units (i, j), push W_f_i away from W_f_j if mu_ij > p_target^2
        # and if both units are active
        # Co-activation mask: (*batch, n_features, n_features)
        zi_zj = z[..., :, None] * z[..., None, :]
        # Compute pairwise centroid differences for all pairs
        # kernel[:, i] - kernel[:, j] for all i, j
        # Using broadcasting: kernel[:, :, None] - kernel[:, None, :]
        # Shape: (input_features, n_features, n_features)
        wfi_minus_wfj = kernel[:, :, None] - kernel[:, None, :]
        # Scale by (p_target^2 - mu_ij) and co-activation
        decorr_scale = (self.p_target**2 - mu)  # (n_features, n_features)
        # We want to sum over j: sum_j [(p_target^2 - mu_ij) * (W_f_i - W_f_j) * z_i * z_j]
        # Result should be (*batch, input_features, n_features)
        # decorr_scale * zi_zj: (*batch, n_features, n_features)
        # wfi_minus_wfj: (input_features, n_features, n_features)
        # We multiply along the last two dims and sum over the last dim
        decorr_weight = decorr_scale * zi_zj  # (*batch, n_features, n_features)
        # Now multiply with wfi_minus_wfj and sum over j (last axis)
        decorr_term = (wfi_minus_wfj * decorr_weight[..., None, :, :]).mean(axis=-1)
        # Result: (*batch, input_features, n_features)
        
        # Combine both terms
        dkernel = self_activity_term + decorr_term
        
        # Average over batch dimensions
        dkernel = dkernel.reshape(-1, *kernel.shape).mean(axis=0)
        
        # Bias update: increase radius if under-active
        dbias = self.p_target - mu_self
        
        return {
            "kernel": dkernel,
            "bias": dbias,
        }
    
    def compute_dw_p(self, params, outs):
        """
        Compute updates for prediction weights using TD learning.
        
        The prediction weights learn to approximate the value function:
        v_t = W_p^T u_t
        
        Updated using TD error: delta = v_{t-1} - (z_t + gamma * v_t)
        
        Args:
            params: Current parameters
            outs: A dictionary, as returned by a forward pass or forward scan
        
        Returns:
            dw_p: Dictionary with kernel and bias updates
        """
        kernel = params["params"]["w_p_kernel"]  # (n_features, n_features)
        mu = params["params"]["mu"]
        mu_self = np.diag(mu)

        u_prev = outs["u_prev"]  # (*batch, n_features)
        td_pred_err_prev = outs["td_pred_err_prev"]  # (*batch, n_features)
        
        # Standard TD update: delta * u
        # u: (*batch, n_features), td_pred_err: (*batch, n_features)
        # We want: u[..., :, None] * td_pred_err[..., None, :]
        # Result: (*batch, n_features, n_features)
        ui_deltaj = u_prev[..., :, None] * td_pred_err_prev[..., None, :]
        
        # Average over batch dimensions
        dkernel = ui_deltaj.reshape(-1, *kernel.shape).mean(axis=0)
        
        # Bias update: adjust threshold based on activity level
        dbias = self.p_target - mu_self
        
        return {
            "kernel": dkernel,
            "bias": dbias,
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
                "w_f_kernel": None,
                "w_f_bias": None,
                "w_p_kernel": None,
                "w_p_bias": None,
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
        dw_p = self.compute_dw_p(
            tmp_params,
            outs,
        )
        
        # Store all updates
        dparams["params"]["mu"] = dmu
        dparams["params"]["w_f_kernel"] = dw_f["kernel"]
        dparams["params"]["w_f_bias"] = dw_f["bias"]
        dparams["params"]["w_p_kernel"] = dw_p["kernel"]
        dparams["params"]["w_p_bias"] = dw_p["bias"]
        
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
        params["params"]["w_f_kernel"] = (
            params["params"]["w_f_kernel"] + lr * dparams["params"]["w_f_kernel"]
        )
        params["params"]["w_f_bias"] = (
            params["params"]["w_f_bias"] + lr * dparams["params"]["w_f_bias"]
        )
        
        # Update prediction weights
        params["params"]["w_p_kernel"] = (
            params["params"]["w_p_kernel"] + lr * dparams["params"]["w_p_kernel"]
        )
        params["params"]["w_p_bias"] = (
            params["params"]["w_p_bias"] + lr * dparams["params"]["w_p_bias"]
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


class CentroidTDModule(CentroidTDBase):
    """
    Recurrent module combining centroid-based forward activation with TD prediction.
    
    Forward pass computes:
    1. y_f: Units activate if input is within distance threshold of centroid
    2. y_p: Units activate if predicted based on discounted history (TD)
    3. z: Final activation is AND of y_f and y_p
    4. u: Updated discounted sum of past activations
    5. v: Value prediction for future activations
    """
    
    def __call__(self, x, u_prev, v_prev, y_p_prev):
        """
        Single time step forward pass.
        
        Args:
            params: Model parameters
            x: Input (*batch_dims, input_features)
            u_prev: Previous discounted sum of activations (*batch_dims, n_features)
            v_prev: Previous value prediction (*batch_dims, n_features), used to generate y_p
            y_p_prev: Prediction of activation, made from the previous step (*batch_dims, n_features) using v_prevprev and v_prev
        
        Returns:
            Dictionary with all activations and values
        """

        # Forward activation: activate if within distance threshold
        kernel_f = self.kernel_f # (input_features, n_features)
        bias_f = self.bias_f  # (n_features,)
        
        # Compute distances from input to each centroid
        # kernel columns are centroids: we want ||x - w_f_i|| for each i
        # x: (*batch, input_features), kernel: (input_features, n_features)
        distances = np.linalg.norm(kernel_f - x[..., None], axis=-2)  # (*batch, n_features)
        y_f = distances < bias_f  # Boolean activation
        
        # Combined activation: AND operation between the input activation and the prediction
        z = np.logical_and(y_f, y_p_prev)
        # Update discounted sum of past activations
        u = self.gamma * u_prev + z.astype(x.dtype)

        # TD-based prediction for next time-step based on past discounted sum
        # # v_t = W_p^T u_t        
        kernel_p = self.kernel_p # (n_features, n_features)
        bias_p = self.bias_p  # (n_features,)
        # u: (*batch, n_features), kernel_p: (n_features[in], n_features[out])
        v = (u[..., None] * kernel_p).sum(axis=-2)  # (*batch, n_features[out])
        
        # TD prediction for next time step
        y_p_next = v_prev - self.gamma * v
        y_p_next = y_p_next + bias_p > 0  # Threshold for binary activation
        
        # Compute TD error for v_prev, computed on u_prev
        td_pred_err_prev = z.astype(x.dtype) - y_p_prev.astype(x.dtype)
        
        return {
            # input/output for current time step
            "x": x,
            "y_f": y_f.astype(x.dtype),
            "z": z.astype(x.dtype),
            # values to be used at next time step
            "u_next": u,
            "v_next": v,
            "y_p_next": y_p_next.astype(x.dtype),
            # values to be used for TD update of previous prediction
            "u_prev": u_prev,
            "td_pred_err_prev": td_pred_err_prev,
        }
    
    def forward_scan(self, x_seq, u_prev_0, v_prev_0, y_p_prev_0, key=None):
        """
        Apply network iteratively over a sequence using jax.lax.scan.
        
        Args:
            x_seq: Input sequence (*batch_dims, time_steps, input_features)
            u0: Initial discounted sum (*batch_dims, n_features)
            v0: Initial value prediction (*batch_dims, n_features)
            key: Random key (not used, but kept for interface compatibility)
        
        Returns:
            Dictionary with all sequences of activations
        """
        def f_scan(carry, input):
            x = input
            u_prev, v_prev, y_p_prev = carry
            outputs = self(x, u_prev, v_prev, y_p_prev)
            
            # Next carry: updated u and v
            u_next = outputs["u_next"]
            v_next = outputs["v_next"]
            y_p_next = outputs["y_p_next"]
            
            return (u_next, v_next, y_p_next), outputs
        
        # Move time axis to first position for scan
        x_seq = np.moveaxis(x_seq, -2, 0)
        
        # Perform scan
        _, outputs_seq = jax.lax.scan(
            f_scan, 
            (u_prev_0, v_prev_0, y_p_prev_0), 
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
            v0: Initial value prediction (*batch_dims, n_features)
            y_p0: Initial prediction (*batch_dims, n_features) for next time step activation
        """
        batch_shape = x.shape[:-1]
        state_shape = (*batch_shape, self.n_features)
        
        # initialize u with the expected value for a discounted sum of bernoulli variables each 
        # with a probability of self.p_target
        u0 = np.ones(state_shape) * self.p_target/(1.0 - self.gamma)
        
        # Initialize v to zeros
        v0 = np.zeros(state_shape)

        # Initialize y_p0 to a bernoulli
        y_p0 = jax.random.bernoulli(key, p=2*self.p_target, shape=state_shape).astype(x.dtype)
        
        return u0, v0, y_p0