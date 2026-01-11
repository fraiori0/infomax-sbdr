"""2D Convolutional Anti-Hebbian TD Module."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple

from .types import Conv2DParams, Conv2DState, Conv2DOutputs


def init_params(
    key: random.PRNGKey,
    kernel_size: Tuple[int, int],
    in_channels: int,
    out_channels: int,
    p_target: float = 0.1,
    init_scale_f: float = 1.0,
) -> Conv2DParams:
    """Initialize Conv2D module parameters."""
    k1, k2 = random.split(key, 2)
    k_h, k_w = kernel_size
    
    W_td = jnp.zeros((k_h, k_w, in_channels, in_channels))
    fan_in = k_h * k_w * in_channels
    W_f = random.normal(k2, (k_h, k_w, in_channels, out_channels)) * jnp.sqrt(
        init_scale_f / fan_in
    )
    b_f = jnp.zeros((out_channels,))
    W_l = jnp.zeros((out_channels, out_channels))
    mu = jnp.ones((out_channels, out_channels)) * p_target**2 + jnp.eye(out_channels) * (
        p_target - p_target**2
    )
    
    return Conv2DParams(W_td=W_td, W_f=W_f, b_f=b_f, W_l=W_l, mu=mu)


def init_state(
    batch_shape: Tuple[int, ...],
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    gamma_f: float,
    gamma_l: float,
    p_target: float,
) -> Conv2DState:
    """Initialize recurrent state."""
    u_x = jnp.ones((*batch_shape, height, width, in_channels)) * p_target / (1.0 - gamma_f)
    u_z = jnp.ones((*batch_shape, height, width, out_channels)) * p_target / (1.0 - gamma_l)
    return Conv2DState(u_x=u_x, u_z=u_z)


def forward_step(
    params: Conv2DParams,
    state: Conv2DState,
    x: jnp.ndarray,
    gamma_f: float,
    gamma_l: float,
    stride: Tuple[int, int] = (1, 1),
    padding: str = "SAME",
) -> Tuple[Conv2DOutputs, Conv2DState]:
    """Single timestep forward pass."""
    u_x_prev, u_z_prev = state.u_x, state.u_z
    
    v_x_prev = jax.lax.conv_general_dilated(
        lhs=u_x_prev,
        rhs=params.W_td,
        window_strides=stride,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    
    h_f = jax.lax.conv_general_dilated(
        lhs=v_x_prev,
        rhs=params.W_f,
        window_strides=stride,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + params.b_f
    
    h_l = u_z_prev @ params.W_l.T
    z = (h_f + h_l > 0).astype(jnp.float32)
    
    u_x = gamma_f * u_x_prev + x
    u_z = gamma_l * u_z_prev + z
    
    v_x = jax.lax.conv_general_dilated(
        lhs=u_x,
        rhs=params.W_td,
        window_strides=stride,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    td_error = x + gamma_f * v_x - v_x_prev
    
    outputs = Conv2DOutputs(
        z=z, u_x=u_x, u_z=u_z, x=x, u_x_prev=u_x_prev, td_error=td_error
    )
    new_state = Conv2DState(u_x=u_x, u_z=u_z)
    
    return outputs, new_state


def forward_scan(
    params: Conv2DParams,
    state: Conv2DState,
    x_seq: jnp.ndarray,
    gamma_f: float,
    gamma_l: float,
    stride: Tuple[int, int] = (1, 1),
    padding: str = "SAME",
) -> Tuple[Conv2DOutputs, Conv2DState]:
    """Process sequence using scan."""
    def scan_fn(carry_state, x_t):
        outputs, new_state = forward_step(
            params, carry_state, x_t, gamma_f, gamma_l, stride, padding
        )
        return new_state, outputs
    
    x_seq_t = jnp.moveaxis(x_seq, -4, 0)
    final_state, outputs_t = jax.lax.scan(scan_fn, state, x_seq_t)
    outputs = jax.tree_util.tree_map(lambda arr: jnp.moveaxis(arr, 0, -4), outputs_t)
    
    return outputs, final_state


def extract_features(outputs: Conv2DOutputs) -> jnp.ndarray:
    """Extract features by pooling over spatial and temporal dimensions."""
    z = outputs.z
    while z.ndim > 2:
        z = z.mean(axis=-2)
    return z
