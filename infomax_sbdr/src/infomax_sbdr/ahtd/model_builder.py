"""Model builder - constructs networks from configurations."""

import jax
import jax.numpy as jnp
from jax import random
from .core.types import AHTDModule
from .core import dense, conv1d, conv2d
from .models.stacking import init_conv2d_stack
from flax.core.frozen_dict import FrozenDict


def build_model(config: dict, key: random.PRNGKey) -> FrozenDict:
    """Build a model from a configuration dictionary (e.g., imported from a TOML file)."""

    hp = []
    p = []
    c = []
    for l in config['layers']:

        hp.append(FrozenDict(l['hyperparams']))
        c.append(FrozenDict(l['config']))
        
        # init params
        if l['config']['type'] == 'dense':
            p.append(dense.init_params(
                key,
                n_input_features=l['config']['n_input_features'],
                n_features=l['config']['n_features'],
                p_target=l['hyperparams']['p_target'],
                init_scale_f=l['config']['init_scale_f'],
            ))
        elif l['config']['type'] == 'conv1d':
            p.append(conv1d.init_params(
                key,
                kernel_size=l['config']['kernel_size'][0],
                in_channels=l['config']['in_channels'],
                out_channels=l['config']['out_channels'],
                p_target=l['hyperparams']['p_target'],
                init_scale_f=l['config']['init_scale_f'],
            ))
        elif l['config']['type'] == 'conv2d':
            p.append(conv2d.init_params(
                key,
                kernel_size=l['config']['kernel_size'],
                in_channels=l['config']['in_channels'],
                out_channels=l['config']['out_channels'],
                p_target=l['hyperparams']['p_target'],
                init_scale_f=l['config']['init_scale_f'],
            ))
        else:
            raise ValueError(f"Unknown module type: {l['config']['type']}")
        
        # refresh key for next layer
        key, _ = random.split(key)

    return AHTDModule(params=tuple(p), hyperparams=tuple(hp), config=tuple(c))


# def build_conv2d_stack(config: dict, key: random.PRNGKey, hyperparams: FrozenDict) -> FrozenDict:
#     """Build stacked Conv2D model."""
#     layer_configs = [
#         {
#             "kernel_size": tuple(layer.kernel_size),
#             "in_channels": layer.in_channels,
#             "out_channels": layer.out_channels,
#             "p_target": hyperparams.p_target,
#             "init_scale_f": layer.init_scale_f,
#         }
#         for layer in config.model.layers
#     ]
#     return init_conv2d_stack(key, layer_configs, hyperparams)


# def build_conv1d_stack(config: Config, key: random.PRNGKey, hyperparams: HyperParams) -> StackedNetwork:
#     """Build stacked Conv1D model."""
#     keys = random.split(key, len(config.model.layers))
#     params_list = []
    
#     for key_i, layer in zip(keys, config.model.layers):
#         params = conv1d.init_params(
#             key_i,
#             kernel_size=layer.kernel_size[0],
#             in_channels=layer.in_channels,
#             out_channels=layer.out_channels,
#             p_target=hyperparams.p_target,
#             init_scale_f=layer.init_scale_f,
#         )
#         params_list.append(params)
    
#     module_types = ["conv1d"] * len(params_list)
#     return StackedNetwork(params_list, module_types, hyperparams)


# def build_dense_stack(config: Config, key: random.PRNGKey, hyperparams: HyperParams) -> StackedNetwork:
#     """Build stacked dense model."""
#     keys = random.split(key, len(config.model.layers))
#     params_list = []
    
#     for key_i, layer in zip(keys, config.model.layers):
#         params = dense.init_params(
#             key_i,
#             n_input_features=layer.in_channels,
#             n_features=layer.out_channels,
#             p_target=hyperparams.p_target,
#             init_scale_f=layer.init_scale_f,
#         )
#         params_list.append(params)
    
#     module_types = ["dense"] * len(params_list)
#     return StackedNetwork(params_list, module_types, hyperparams)


# def build_single_dense(config: Config, key: random.PRNGKey, hyperparams: HyperParams) -> StackedNetwork:
#     """Build single dense layer model."""
#     assert len(config.model.layers) == 1
#     layer = config.model.layers[0]
#     params = dense.init_params(
#         key,
#         n_input_features=layer.in_channels,
#         n_features=layer.out_channels,
#         p_target=hyperparams.p_target,
#         init_scale_f=layer.init_scale_f,
#     )
#     return StackedNetwork([params], ["dense"], hyperparams)


# def get_model_info(network: StackedNetwork) -> str:
#     """Get model information string."""
#     lines = ["=" * 60, "Model Architecture", "=" * 60]
#     lines.append(f"Number of layers: {network.n_layers}")
#     lines.append("")
    
#     total_params = 0
    
#     for i, (params, module_type) in enumerate(zip(network.params_list, network.module_types)):
#         lines.append(f"Layer {i + 1}: {module_type}")
#         lines.append("-" * 40)
        
#         layer_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
#         total_params += layer_params
        
#         if module_type == "dense":
#             lines.append(f"  Input features: {params.W_f.shape[0]}")
#             lines.append(f"  Output features: {params.W_f.shape[1]}")
#         elif module_type == "conv1d":
#             k, in_c, out_c = params.W_f.shape
#             lines.append(f"  Kernel size: {k}")
#             lines.append(f"  In channels: {in_c}")
#             lines.append(f"  Out channels: {out_c}")
#         elif module_type == "conv2d":
#             kh, kw, in_c, out_c = params.W_f.shape
#             lines.append(f"  Kernel size: {kh}x{kw}")
#             lines.append(f"  In channels: {in_c}")
#             lines.append(f"  Out channels: {out_c}")
        
#         lines.append(f"  Parameters: {layer_params:,}")
#         lines.append("")
    
#     lines.extend(["=" * 60, f"Total parameters: {total_params:,}", "=" * 60])
    
#     hp = network.hyperparams
#     lines.extend([
#         "",
#         "Hyperparameters:",
#         f"  gamma_f: {hp.gamma_f}",
#         f"  gamma_l: {hp.gamma_l}",
#         f"  p_target: {hp.p_target}",
#         f"  momentum: {hp.momentum}",
#         f"  lr: {hp.lr}",
#         "=" * 60,
#     ])
    
#     return "\n".join(lines)
