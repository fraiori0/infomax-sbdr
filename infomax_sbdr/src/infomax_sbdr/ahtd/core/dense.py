"""Dense Anti-Hebbian TD Module."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple

from .types import DenseParams, DenseState, DenseOutputs
from flax.core.frozen_dict import FrozenDict


def init_params(
    key: random.PRNGKey,
    n_input_features: int,
    n_features: int,
    p_target: float = 0.1,
    init_scale_f: float = 1.0,
) -> FrozenDict:
    """Initialize dense module parameters."""
    k1, k2 = random.split(key, 2)
    
    # (input, output)
    W_td = jnp.zeros((n_input_features, n_input_features))
    W_f = random.normal(k2, (n_input_features, n_features)) * jnp.sqrt(
        init_scale_f / n_input_features
    )
    b_f = jnp.zeros((n_features,))
    W_l = jnp.zeros((n_features, n_features))
    mu = jnp.ones((n_features, n_features)) * p_target**2 + jnp.eye(n_features) * (
        p_target - p_target**2
    )
    
    return DenseParams(W_td=W_td, W_f=W_f, b_f=b_f, W_l=W_l, mu=mu)


def init_state(
    batch_shape: Tuple[int, ...],
    n_input_features: int,
    n_features: int,
    gamma_f: float,
    gamma_l: float,
    p_target: float,
) -> FrozenDict:
    """Initialize recurrent state."""
    u_x = jnp.ones((*batch_shape, n_input_features)) * p_target / (1.0 - gamma_f)
    u_z = jnp.ones((*batch_shape, n_features)) * p_target / (1.0 - gamma_l)
    return DenseState(u_x=u_x, u_z=u_z)


def forward_step(
    params: FrozenDict,
    state: FrozenDict,
    x: jnp.ndarray,
    gamma_f: float,
    gamma_l: float,
) -> Tuple[FrozenDict, FrozenDict]:
    """Single timestep forward pass."""
    u_x_prev, u_z_prev = state["u_x"], state["u_z"]
    
    v_x_prev = u_x_prev @ params["W_td"]
    # v_x_prev = u_z_prev @ params["W_td"]
    # h_f = v_x_prev @ params["W_f"] + params["b_f"]
    h_f = v_x_prev @ params["W_f"] + params["b_f"]
    h_l = u_z_prev @ params["W_l"]
    z = (h_f + h_l > 0).astype(jnp.float32)
    
    u_x = gamma_f * u_x_prev + x
    u_z = gamma_l * u_z_prev + z
    
    v_x = u_x @ params["W_td"]
    # v_x = u_z @ params["W_td"]
    td_error = x + gamma_f * v_x - v_x_prev
    
    outputs = DenseOutputs(
        z=z, u_x=u_x, u_z=u_z, x=x, u_x_prev=u_x_prev, u_z_prev=u_z_prev, td_error=td_error,
    )
    new_state = DenseState(u_x=u_x, u_z=u_z)
    
    return outputs, new_state


def forward_scan(
    params: FrozenDict,
    state: FrozenDict,
    x_seq: jnp.ndarray,
    gamma_f: float,
    gamma_l: float,
) -> Tuple[FrozenDict, FrozenDict]:
    """Process sequence using scan."""
    def scan_fn(carry_state, x_t):
        outputs, new_state = forward_step(params, carry_state, x_t, gamma_f, gamma_l)
        return new_state, outputs
    
    x_seq_t = jnp.moveaxis(x_seq, -2, 0)
    final_state, outputs_t = jax.lax.scan(scan_fn, state, x_seq_t)
    outputs = jax.tree_util.tree_map(lambda arr: jnp.moveaxis(arr, 0, -2), outputs_t)
    
    return outputs, final_state


def extract_features(outputs: FrozenDict) -> jnp.ndarray:
    """Extract features by averaging over time."""
    return outputs.z.mean(axis=-2)
