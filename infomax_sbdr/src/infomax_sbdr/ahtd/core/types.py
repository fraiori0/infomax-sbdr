"""Type definitions for Anti-Hebbian TD modules."""

from typing import NamedTuple, Tuple, Union, List
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


# Hyperparameters
def HyperParams(
        gamma_f: float = 0.9,
        gamma_l: float = 0.8,
        p_target: float = 0.1,
        momentum: float = 0.95,
        lr: float = 0.1,
    ):
    return FrozenDict({
        'gamma_f': gamma_f,
        'gamma_l': gamma_l,
        'p_target': p_target,
        'momentum': momentum,
        'lr': lr,
    })


# Dense module types
def DenseParams(
        W_f_td: jnp.ndarray,  # (n_input_features, n_input_features)
        b_f_td: jnp.ndarray,  # (n_input_features,)
        W_l_td: jnp.ndarray,  # (n_features, n_input_features)
        b_l_td: jnp.ndarray,  # (n_input_features,)
        W_f: jnp.ndarray,   # (n_input_features, n_features)
        b_f: jnp.ndarray,   # (n_features,)
        W_l: jnp.ndarray,   # (n_features, n_features)
        mu: jnp.ndarray,    # (n_features, n_features)
    ):
    return FrozenDict({
        'W_f_td': W_f_td,
        'b_f_td': b_f_td,
        'W_l_td': W_l_td,
        'b_l_td': b_l_td,
        'W_f': W_f,
        'b_f': b_f,
        'W_l': W_l,
        'mu': mu,
    })


def DenseState(
        u_x: jnp.ndarray,
        u_z: jnp.ndarray,
        u_e: jnp.ndarray,
    ):
    return FrozenDict({
        'u_x': u_x,
        'u_z': u_z,
        'u_e': u_e,
    })


def DenseOutputs(
        z: jnp.ndarray,
        u_x: jnp.ndarray,
        u_z: jnp.ndarray,
        input_f: jnp.ndarray,
        input_l: jnp.ndarray,
        u_x_prev: jnp.ndarray,
        u_z_prev: jnp.ndarray,
        td_error_f: jnp.ndarray,
        td_error_l: jnp.ndarray,
    ):
    return FrozenDict({
        'z': z,
        'u_x': u_x,
        'u_z': u_z,
        'input_f': input_f,
        'input_l': input_l,
        'u_x_prev': u_x_prev,
        'u_z_prev': u_z_prev,
        'td_error_f': td_error_f,
        'td_error_l': td_error_l,
    })


# Conv1D module types
def Conv1DParams(
        W_td: jnp.ndarray,  # (kernel_size, in_channels, in_channels)
        W_f: jnp.ndarray,   # (kernel_size, in_channels, out_channels)
        b_f: jnp.ndarray,   # (out_channels,)
        W_l: jnp.ndarray,   # (out_channels, out_channels)
        mu: jnp.ndarray,    # (out_channels, out_channels)
    ):
    return FrozenDict({
        'W_td': W_td,
        'W_f': W_f,
        'b_f': b_f,
        'W_l': W_l,
        'mu': mu,
    })


def Conv1DState(
        u_x: jnp.ndarray,
        u_z: jnp.ndarray,
    ):
    return FrozenDict({
        'u_x': u_x,
        'u_z': u_z,
    })    


def Conv1DOutputs(
        z: jnp.ndarray,
        u_x: jnp.ndarray,
        u_z: jnp.ndarray,
        x: jnp.ndarray,
        u_x_prev: jnp.ndarray,
        td_error: jnp.ndarray,
    ):
    return FrozenDict({
        'z': z,
        'u_x': u_x,
        'u_z': u_z,
        'x': x,
        'u_x_prev': u_x_prev,
        'td_error': td_error
    })    


# Conv2D module types
def Conv2DParams(
        W_td: jnp.ndarray,  # (k_h, k_w, in_channels, in_channels)
        W_f: jnp.ndarray,   # (k_h, k_w, in_channels, out_channels)
        b_f: jnp.ndarray,   # (out_channels,)
        W_l: jnp.ndarray,   # (out_channels, out_channels)
        mu: jnp.ndarray,    # (out_channels, out_channels)
    ):
    return FrozenDict({
        'W_td': W_td,
        'W_f': W_f,
        'b_f': b_f,
        'W_l': W_l,
        'mu': mu,
    })


def Conv2DState(
        u_x: jnp.ndarray,
        u_z: jnp.ndarray,
    ):
    return FrozenDict({
        'u_x': u_x,
        'u_z': u_z,
    })


def Conv2DOutputs(
        z: jnp.ndarray,
        u_x: jnp.ndarray,
        u_z: jnp.ndarray,
        x: jnp.ndarray,
        u_x_prev: jnp.ndarray,
        td_error: jnp.ndarray,
    ):
    return FrozenDict({
        'z': z,
        'u_x': u_x,
        'u_z': u_z,
        'x': x,
        'u_x_prev': u_x_prev,
        'td_error': td_error
    })



# Module configuration
def DenseConfig(
        n_input_features: int,
        n_features: int,
        init_scale_f: float = 1.0,
    ):
    return FrozenDict({
        'n_input_features': n_input_features,
        'n_features': n_features,
        'init_scale_f': init_scale_f,
        'type': 'dense',
    })


def Conv1DConfig(
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        init_scale_f: float = 1.0,
    ):
    return FrozenDict({
        'kernel_size': kernel_size,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'init_scale_f': init_scale_f,
        'type': 'conv1d',
    })


def Conv2DConfig(
        kernel_size: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        init_scale_f: float = 1.0,
    ):
    return FrozenDict({
        'kernel_size': kernel_size,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'stride': stride,
        'padding': padding,
        'init_scale_f': init_scale_f,
        'type': 'conv2d',
    })


def AHTDModule(
        params: Tuple[FrozenDict],
        hyperparams: Tuple[FrozenDict],
        config: Tuple[FrozenDict],  
    ):
    return FrozenDict({
        'params': params,
        'hyperparams': hyperparams,
        'config': config,
        # 'n_modules': len(params),
    })
