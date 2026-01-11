"""1D Convolutional Anti-Hebbian TD Module."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple

from .types import Conv1DParams, Conv1DState, Conv1DOutputs


def init_params(
    key: random.PRNGKey,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    p_target: float = 0.1,
    init_scale_f: float = 1.0,
) -> Conv1DParams:
    """Initialize Conv1D module parameters."""
    k1, k2 = random.split(key, 2)
    
    W_td = jnp.zeros((kernel_size, in_channels, in_channels))
    fan_in = kernel_size * in_channels
    W_f = random.normal(k2, (kernel_size, in_channels, out_channels)) * jnp.sqrt(
        init_scale_f / fan_in
    )
    b_f = jnp.zeros((out_channels,))
    W_l = jnp.zeros((out_channels, out_channels))
    mu = jnp.ones((out_channels, out_channels)) * p_target**2 + jnp.eye(out_channels) * (
        p_target - p_target**2
    )
    
    return Conv1DParams(W_td=W_td, W_f=W_f, b_f=b_f, W_l=W_l, mu=mu)


def init_state(
    batch_shape: Tuple[int, ...],
    time_length: int,
    in_channels: int,
    out_channels: int,
    gamma_f: float,
    gamma_l: float,
    p_target: float,
) -> Conv1DState:
    """Initialize recurrent state."""
    u_x = jnp.ones((*batch_shape, time_length, in_channels)) * p_target / (1.0 - gamma_f)
    u_z = jnp.ones((*batch_shape, time_length, out_channels)) * p_target / (1.0 - gamma_l)
    return Conv1DState(u_x=u_x, u_z=u_z)


def forward_step(
    params: Conv1DParams,
    state: Conv1DState,
    x: jnp.ndarray,
    gamma_f: float,
    gamma_l: float,
) -> Tuple[Conv1DOutputs, Conv1DState]:
    """Single timestep forward pass."""
    u_x_prev, u_z_prev = state.u_x, state.u_z
    
    v_x_prev = jax.lax.conv_general_dilated(
        lhs=u_x_prev,
        rhs=params.W_td,
        window_strides=(1,),
        padding="SAME",
        dimension_numbers=("NTC", "TIO", "NTC"),
    )
    
    h_f = jax.lax.conv_general_dilated(
        lhs=v_x_prev,
        rhs=params.W_f,
        window_strides=(1,),
        padding="SAME",
        dimension_numbers=("NTC", "TIO", "NTC"),
    ) + params.b_f
    
    h_l = u_z_prev @ params.W_l.T
    z = (h_f + h_l > 0).astype(jnp.float32)
    
    u_x = gamma_f * u_x_prev + x
    u_z = gamma_l * u_z_prev + z
    
    v_x = jax.lax.conv_general_dilated(
        lhs=u_x,
        rhs=params.W_td,
        window_strides=(1,),
        padding="SAME",
        dimension_numbers=("NTC", "TIO", "NTC"),
    )
    td_error = x + gamma_f * v_x - v_x_prev
    
    outputs = Conv1DOutputs(
        z=z, u_x=u_x, u_z=u_z, x=x, u_x_prev=u_x_prev, td_error=td_error
    )
    new_state = Conv1DState(u_x=u_x, u_z=u_z)
    
    return outputs, new_state


def forward_scan(
    params: Conv1DParams,
    state: Conv1DState,
    x_seq: jnp.ndarray,
    gamma_f: float,
    gamma_l: float,
) -> Tuple[Conv1DOutputs, Conv1DState]:
    """Process sequence using scan."""
    def scan_fn(carry_state, x_t):
        outputs, new_state = forward_step(params, carry_state, x_t, gamma_f, gamma_l)
        return new_state, outputs
    
    x_seq_t = jnp.moveaxis(x_seq, -3, 0)
    final_state, outputs_t = jax.lax.scan(scan_fn, state, x_seq_t)
    outputs = jax.tree_util.tree_map(lambda arr: jnp.moveaxis(arr, 0, -3), outputs_t)
    
    return outputs, final_state


def extract_features(outputs: Conv1DOutputs) -> jnp.ndarray:
    """Extract features by pooling over temporal dimension."""
    return outputs.z.mean(axis=-2)
