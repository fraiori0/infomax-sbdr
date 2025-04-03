import jax
import jax.numpy as np
from jax import jit, grad, vmap
import sparse_distributed_memory.binary_comparisons as bn


def single_classifier_layer(params, x):
    # single layer with sigmoid activation
    x = np.dot(params["w"], x) + params["b"]
    return jax.nn.log_softmax(x)


def simple_classifier(params, y):
    y = y.reshape(-1)
    y = single_classifier_layer(params, y)
    return y


def gen_params_simple_classifier(key, in_features, out_features, scale=0.01):
    key_w, _ = jax.random.split(key, 2)
    params = {
        "w": scale * jax.random.normal(key_w, shape=(out_features, in_features)),
        "b": np.zeros(out_features),
    }

    return params


def multilayer_classifier(params, x):
    x = x.reshape(-1)
    for layer in params[:-1]:
        x = np.dot(layer["w"], x) + layer["b"]
        x = jax.nn.gelu(x)
    x = np.dot(params[-1]["w"], x) + params[-1]["b"]
    return jax.nn.log_softmax(x)


def gen_params_multilayer_classifier(key, layer_sizes, scale=0.01):
    params = []
    for i in range(len(layer_sizes) - 1):
        key_w, _ = jax.random.split(key, 2)
        params.append(
            {
                "w": scale
                * jax.random.normal(
                    key_w, shape=(layer_sizes[i + 1], layer_sizes[i])
                ),
                "b": np.zeros(layer_sizes[i + 1]),
            }
        )
    return params


def info_sdr_classifier(params, y):
    # here the params are a set of SDRs, one representing each label
    s = bn.expected_custom_index_3(params["l"], y[None, :])
    return jax.nn.log_softmax(s)


def gen_params_info_sdr_classifier(key, in_features, out_features, scale=0.01):
    key_l, _ = jax.random.split(key, 2)
    params = {"l": np.zeros(shape=(out_features, in_features)), }
    return params
