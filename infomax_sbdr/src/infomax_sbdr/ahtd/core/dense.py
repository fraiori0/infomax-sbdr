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
    W_f_td = jnp.zeros((n_input_features, n_input_features))
    b_f_td = jnp.zeros((n_input_features,))
    W_l_td = jnp.zeros((n_features, n_input_features))
    b_l_td = jnp.zeros((n_input_features,))
    W_f = random.normal(k2, (n_input_features, n_features)) * jnp.sqrt(
        init_scale_f / n_input_features
    )
    b_f = jnp.zeros((n_features,))
    W_l = jnp.zeros((n_features, n_features))
    mu = jnp.ones((n_features, n_features)) * p_target**2 + jnp.eye(n_features) * (
        p_target - p_target**2
    )
    
    return DenseParams(W_f_td=W_f_td, b_f_td=b_f_td, W_l_td=W_l_td, b_l_td=b_l_td, W_f=W_f, b_f=b_f, W_l=W_l, mu=mu)


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
    u_e = jnp.zeros((*batch_shape, n_input_features))
    return DenseState(u_x=u_x, u_z=u_z, u_e=u_e)


def forward_step(
    params: FrozenDict,
    state: FrozenDict,
    x: jnp.ndarray,
    gamma_f: float,
    gamma_l: float,
) -> Tuple[FrozenDict, FrozenDict]:
    """Single timestep forward pass."""
    u_x_prev, u_z_prev, u_e_prev = state["u_x"], state["u_z"], state["u_e"]
    
    # # # Compute TD predictions
    # Compute centered input
    # x_flat = x.reshape(-1, x.shape[-1])
    # d_x = x - x_flat.mean(axis=0)
    # Update u_x
    u_x = gamma_f * u_x_prev + x
    # Compute predictions using trace of x
    v_x_f_prev = u_x_prev @ params["W_f_td"] + params["b_f_td"]
    v_x_f = u_x @ params["W_f_td"] + params["b_f_td"]
    # Compute prediction using trace of z
    v_x_l_prev = u_z_prev @ params["W_l_td"] + params["b_l_td"]
    # # # Compute TD error
    # standard td_error
    # td_error = x + gamma_f * v_x - v_x_prev
    # # weight td error for sparse events, increasing the weight of activate input features
    # c = x/0.2 + 1
    # td_error = c * td_error
    # Compute td error considering a centering decomposition
    td_error_f = x + gamma_f * v_x_f - v_x_f_prev
    u_e = gamma_l * u_e_prev + td_error_f

    # # # Compute Forward Pass
    # h_f = v_x_prev @ params["W_f"] + params["b_f"]
    # similar to predictive coding: input is zero if predictions were correct
    input_f = v_x_f# td_error_f
    input_l = u_z_prev
    h_f = input_f @ params["W_f"] + params["b_f"]
    h_l = input_l @ params["W_l"]
    z = (h_f + h_l > 0).astype(jnp.float32)

    # Update recurrent state
    u_z = gamma_l * u_z_prev + z
    # Update Td prediction
    v_x_l = u_z @ params["W_l_td"]
    td_error_l = x + gamma_l * v_x_l - v_x_l_prev
    
    outputs = DenseOutputs(
        z=z, u_x=u_x, u_z=u_z, input_f=input_f, input_l=input_l, u_x_prev=u_x_prev, u_z_prev=u_z_prev, td_error_f=td_error_f, td_error_l=td_error_l,
    )
    new_state = DenseState(u_x=u_x, u_z=u_z, u_e=u_e)
    
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
    z = outputs["z"]
    # z = z/(z.mean(axis=-1, keepdims=True)+1e-8)
    z = z.mean(axis=-2)
    # return outputs["u_z"][..., -1, :]
    return z
