"""Module stacking and composition."""

import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from functools import partial
from typing import List, Tuple, Callable, Union

from infomax_sbdr.ahtd.core.types import *
from infomax_sbdr.ahtd.core import dense, conv1d, conv2d
from infomax_sbdr.ahtd.learning import updates
from flax.core.frozen_dict import FrozenDict

    
def get_module_forward(module_type: str) -> Callable:
    if module_type == "dense":
        return dense.forward_scan
    elif module_type == "conv1d":
        return conv1d.forward_scan
    elif module_type == "conv2d":
        return conv2d.forward_scan
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
def get_init_state_fn(module_type: str) -> Callable:
    if module_type == "dense":
        return dense.init_state
    elif module_type == "conv1d":
        return conv1d.init_state
    elif module_type == "conv2d":
        return conv2d.init_state
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
def get_update_fn(module_type: str) -> Tuple[Callable, Callable]:
    if module_type == "dense":
        return updates.compute_dense_updates, updates.apply_dense_updates
    elif module_type in ["conv1d", "conv2d"]:
        return updates.compute_conv_updates, updates.apply_conv_updates
    else:
        raise ValueError(f"Unknown module type: {module_type}")

# @partial(jit, static_argnums=(2, 3))
def init_state_from_input(
    x: jnp.ndarray,
    params: Tuple[FrozenDict],
    hyperparams: Tuple[FrozenDict],
    config: Tuple[FrozenDict],
) -> FrozenDict:
    """Initialize module state from input tensor, for each module in the stack."""

    states = []
    for i in range(len(params)):

        module_type = config[i]["type"]

        if module_type == "dense":
            batch_shape = x.shape[:-2]
            time_length = x.shape[-2]
            n_input = x.shape[-1]
            n_output = params[i]["W_f"].shape[-1]
            states.append(dense.init_state(
                batch_shape, n_input, n_output, hyperparams[i]["gamma_f"], hyperparams[i]["gamma_l"], hyperparams[i]["p_target"]
            ))
            # set "input" for next layer
            x = jnp.empty((*states[-1]["u_z"].shape[:-1], time_length, states[-1]["u_z"].shape[-1]))
        
        elif module_type == "conv1d":
            batch_shape = x.shape[:-3]
            time_length = x.shape[-2]
            in_channels = x.shape[-1]
            out_channels = params[i]["W_f"].shape[-1]
            states.append(conv1d.init_state(
                batch_shape, time_length, in_channels, out_channels,
                hyperparams[i]["gamma_f"], hyperparams[i]["gamma_l"], hyperparams[i]["p_target"]
            ))
            # set "input" for next layer
            x = states[-1]["u_z"].copy()
        
        elif module_type == "conv2d":
            batch_shape = x.shape[:-4]
            height, width = x.shape[-3], x.shape[-2]
            in_channels = x.shape[-1]
            out_channels = params[i]["W_f"].shape[-1]
            states.append(conv2d.init_state(
                batch_shape, height, width, in_channels, out_channels,
                hyperparams[i]["gamma_f"], hyperparams[i]["gamma_l"], hyperparams[i]["p_target"]
            ))
            # set "input" for next layer
            x = states[-1]["u_z"].copy()
        
        else:
            raise ValueError(f"Unknown module type: {module_type}")
        
    return tuple(states)

@partial(jit, static_argnums=(3, 4))
def forward_stack(
    state: Tuple[FrozenDict],
    x_seq: jnp.ndarray,
    params: Tuple[FrozenDict],
    hyperparams: Tuple[FrozenDict],
    config: Tuple[FrozenDict],
) -> Tuple[FrozenDict]:
    """Forward pass through entire stack."""
    h = x_seq
    outs = []
    for layer_idx in range(len(params)):
        
        module_type = config[layer_idx]["type"]
        module_hp = hyperparams[layer_idx]
        module_p = params[layer_idx]
        module_state = state[layer_idx]
        forward_fn = get_module_forward(module_type)
        
        outputs, _ = forward_fn(module_p, module_state, h, module_hp["gamma_f"], module_hp["gamma_l"])
        
        h = outputs["z"]
        
        outs.append(outputs)
    
    return tuple(outs)

@partial(jit, static_argnums=(2, 3, 4))
def update_stack(
    outputs: Tuple[FrozenDict],
    params: Tuple[FrozenDict],
    hyperparams: Tuple[FrozenDict],
    config: Tuple[FrozenDict],
    lr: float = None,
) -> FrozenDict:
    """Perform one learning update on all layers."""
    updated_params_list = []
    
    for layer_idx in range(len(params)):

        module_type = config[layer_idx]["type"]
        module_hp = hyperparams[layer_idx]
        module_p = params[layer_idx]
        out = outputs[layer_idx]

        if lr is None:
            lr = module_hp["lr"]

        compute_updates_fn, apply_updates_fn = get_update_fn(module_type)
        delta_p = compute_updates_fn(module_p, out, module_hp["p_target"], module_hp["momentum"])
        updated_params = apply_updates_fn(module_p, delta_p, lr)
        updated_params_list.append(updated_params)
    

    return tuple(updated_params_list)

@partial(jit, static_argnums=(3, 4, 5))
def extract_features(
    state: FrozenDict,
    x_seq: jnp.ndarray,
    params: Tuple[FrozenDict],
    hyperparams: Tuple[FrozenDict],
    config: Tuple[FrozenDict],
    layer_idxs: Tuple[int] = (-1,),
) -> jnp.ndarray:
    
    """Extract features from a specific layer."""
    outs = forward_stack(state, x_seq, params, hyperparams, config)

    y = []
    for l in layer_idxs:
        out = outs[l]
        module_type = config[l]['type']
        
        if module_type == "dense":
            y.append(dense.extract_features(out))
        elif module_type == "conv1d":
            y.append(conv1d.extract_features(out))
        elif module_type == "conv2d":
            y.append(conv2d.extract_features(out))
        else:
            raise ValueError(f"Unknown module type: {module_type}")
        
    return jnp.concatenate(y, axis=-1)


def init_conv2d_stack(
    key: random.PRNGKey,
    layer_configs: List[dict],
    hyperparams: HyperParams,
) -> FrozenDict:
    pass
    # """Initialize a stack of Conv2D modules."""
    # keys = random.split(key, len(layer_configs))
    # params_list = []
    
    # for key_i, config in zip(keys, layer_configs):
    #     params = conv2d.init_params(
    #         key_i,
    #         kernel_size=config["kernel_size"],
    #         in_channels=config["in_channels"],
    #         out_channels=config["out_channels"],
    #         p_target=config.get("p_target", hyperparams.p_target),
    #         init_scale_f=config.get("init_scale_f", 1.0),
    #     )
    #     params_list.append(params)
    
    # module_types = ["conv2d"] * len(layer_configs)
    
    # return StackedNetwork(
    #     params_list=params_list,
    #     module_types=module_types,
    #     hyperparams=hyperparams,
    # )
