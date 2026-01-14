"""Learning rules for Anti-Hebbian TD modules."""

import jax.numpy as jnp
from typing import Union

from infomax_sbdr.ahtd.core.types import (
    DenseParams, DenseOutputs,
    Conv1DParams, Conv1DOutputs,
    Conv2DParams, Conv2DOutputs,
)
from flax.core.frozen_dict import FrozenDict


def compute_dense_updates(
    params: FrozenDict,
    outputs: FrozenDict,
    p_target: float,
    momentum: float,
) -> FrozenDict:
    """Compute parameter updates for dense module."""
    z = outputs["z"]
    input_f = outputs["input_f"]
    input_l = outputs["input_l"]
    u_x_prev = outputs["u_x_prev"]
    u_z_prev = outputs["u_z_prev"]
    td_error_f = outputs["td_error_f"]
    td_error_l = outputs["td_error_l"]
    
    # Flatten batch and time
    z_flat = z.reshape(-1, z.shape[-1])
    input_f_flat = input_f.reshape(-1, input_f.shape[-1])
    input_l_flat = input_l.reshape(-1, input_l.shape[-1])
    u_x_prev_flat = u_x_prev.reshape(-1, u_x_prev.shape[-1])
    u_z_prev_flat = u_z_prev.reshape(-1, u_z_prev.shape[-1])
    td_error_f_flat = td_error_f.reshape(-1, td_error_f.shape[-1])
    td_error_l_flat = td_error_l.reshape(-1, td_error_l.shape[-1])
    n_samples = z_flat.shape[0]
    
    # Co-activation statistics
    z_outer = (z_flat[..., :, None] * z_flat[..., None, :]).mean(axis=0)
    delta_mu = (1.0 - momentum) * (z_outer - params["mu"])
    mu_updated = params["mu"] + delta_mu
    mu_diag = jnp.diag(mu_updated)
    
    # Forward weights
    # pulls kernel toward input if unit is active
    # proportionally to the difference between the target and the actual co-activation
    xi_zj = (input_f_flat[..., :, None] - params["W_f"]) * z_flat[..., None, :]
    weight = (p_target**2 - mu_diag**2)
    delta_W_f = xi_zj * weight
    delta_W_f = delta_W_f.reshape(-1, *params["W_f"].shape).mean(axis=0)
    delta_b_f = p_target - mu_diag
    
    # Lateral weights
    # Pull kernel, update only active units
    zi_zj = (input_l_flat[..., :, None] - params["W_l"]) * z_flat[..., None, :]
    weight = p_target**2 - mu_updated
    delta_W_l = zi_zj * weight
    delta_W_l = delta_W_l.reshape(-1, *params["W_l"].shape).mean(axis=0)
    # delta_W_l = weight * z_outer
    
    # # # TD weights
    # Forward
    delta_W_f_td = u_x_prev_flat[..., :, None] * td_error_f_flat[..., None, :]
    delta_W_f_td = delta_W_f_td.mean(axis=0)
    delta_b_f_td = td_error_f_flat.mean(axis=0)
    # Lateral
    delta_W_l_td = u_z_prev_flat[..., :, None] * td_error_l_flat[..., None, :]
    delta_W_l_td = delta_W_l_td.mean(axis=0)
    delta_b_l_td = td_error_l_flat.mean(axis=0)
    
    
    return DenseParams(
        W_f_td=delta_W_f_td,
        b_f_td=delta_b_f_td,
        W_l_td=delta_W_l_td,
        b_l_td=delta_b_l_td,
        W_f=delta_W_f,
        b_f=delta_b_f,
        W_l=delta_W_l,
        mu=delta_mu,
    )


def apply_dense_updates(
    params: FrozenDict,
    delta_params: FrozenDict,
    lr: float,
) -> FrozenDict:
    """Apply updates with constraints."""
    W_f_td = params["W_f_td"] + lr * delta_params["W_f_td"]
    b_f_td = params["b_f_td"]*0 #+ lr * delta_params["b_f_td"]
    W_l_td = params["W_l_td"] + lr * delta_params["W_l_td"]
    b_l_td = params["b_l_td"]*0 #+ lr * delta_params["b_l_td"]
    W_f = params["W_f"] + lr * delta_params["W_f"]
    b_f = params["b_f"] + lr * delta_params["b_f"]
    W_l = params["W_l"] + lr * delta_params["W_l"]
    mu = params["mu"] + delta_params["mu"]
    
    # Zero diagonals
    W_l = W_l - jnp.diag(jnp.diag(W_l))
    W_f_td = W_f_td - jnp.diag(jnp.diag(W_f_td))
    
    
    return DenseParams(W_f_td=W_f_td, b_f_td=b_f_td, W_l_td=W_l_td, b_l_td=b_l_td, W_f=W_f, b_f=b_f, W_l=W_l, mu=mu)


def compute_conv_updates(
    params: FrozenDict,
    outputs: FrozenDict,
    p_target: float,
    momentum: float,
) -> FrozenDict:
    """Compute parameter updates for convolutional modules."""
    z = outputs["z"]
    x = outputs["x"]
    u_x_prev = outputs["u_x_prev"]
    td_error = outputs["td_error"]
    
    # Flatten all dimensions except channels
    z_flat = z.reshape(-1, z.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])
    u_x_prev_flat = u_x_prev.reshape(-1, u_x_prev.shape[-1])
    td_error_flat = td_error.reshape(-1, td_error.shape[-1])
    n_samples = z_flat.shape[0]
    
    # Co-activation statistics
    z_outer = (z_flat.T @ z_flat) / n_samples
    delta_mu = (1.0 - momentum) * (z_outer - params["mu"])
    mu_updated = params["mu"] + delta_mu
    mu_diag = jnp.diag(mu_updated)
    
    # Forward weights (simplified: put gradient at kernel center)
    weight = (p_target**2 - mu_diag**2) * z_flat
    x_z_corr = (x_flat.T @ weight) / n_samples  # (in_channels, out_channels)
    
    kernel_shape = params["W_f"].shape
    delta_W_f = jnp.zeros_like(params["W_f"])
    
    if len(kernel_shape) == 3:  # Conv1D
        center_idx = kernel_shape[0] // 2
        delta_W_f = delta_W_f.at[center_idx].set(x_z_corr)
    elif len(kernel_shape) == 4:  # Conv2D
        center_h, center_w = kernel_shape[0] // 2, kernel_shape[1] // 2
        delta_W_f = delta_W_f.at[center_h, center_w].set(x_z_corr)
    
    delta_b_f = p_target - mu_diag
    
    # Lateral weights
    weight_matrix = p_target**2 - mu_updated
    delta_W_l = weight_matrix * z_outer
    
    # TD weights
    td_corr = (td_error_flat.T @ u_x_prev_flat) / n_samples  # (in_channels, in_channels)
    delta_W_td = jnp.zeros_like(params["W_td"])
    
    if len(kernel_shape) == 3:  # Conv1D
        center_idx = kernel_shape[0] // 2
        delta_W_td = delta_W_td.at[center_idx].set(td_corr)
    elif len(kernel_shape) == 4:  # Conv2D
        center_h, center_w = kernel_shape[0] // 2, kernel_shape[1] // 2
        delta_W_td = delta_W_td.at[center_h, center_w].set(td_corr)
    
    ParamsType = type(params)
    return ParamsType(
        W_td=delta_W_td,
        W_f=delta_W_f,
        b_f=delta_b_f,
        W_l=delta_W_l,
        mu=delta_mu,
    )


def apply_conv_updates(
    params: FrozenDict,
    delta_params: FrozenDict,
    lr: float,
) -> FrozenDict:
    """Apply updates for convolutional modules."""
    W_td = params["W_td"] + lr * delta_params["W_td"]
    W_f = params["W_f"] + lr * delta_params["W_f"]
    b_f = params["b_f"] + lr * delta_params["b_f"]
    W_l = params["W_l"] + lr * delta_params["W_l"]
    mu = params["mu"] + delta_params["mu"]
    
    # Zero diagonal of lateral
    W_l = W_l - jnp.diag(jnp.diag(W_l))
    
    ParamsType = type(params)
    return ParamsType(W_td=W_td, W_f=W_f, b_f=b_f, W_l=W_l, mu=mu)
