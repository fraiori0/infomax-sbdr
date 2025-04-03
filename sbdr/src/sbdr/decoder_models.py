import jax
import jax.numpy as np
from jax import jit, grad, vmap


def single_decoder_layer(params, x):
    # single layer with sigmoid activation
    x = np.dot(params["w"], x) + params["b"]
    return jax.nn.tanh(x)


def simple_decoder(params, y):
    y = y.reshape(-1)
    y = single_decoder_layer(params, y)
    return y


def gen_params_simple_decoder(key, in_features, out_features, scale=0.01):
    key_w, _ = jax.random.split(key, 2)
    params = {
        "w": scale * jax.random.normal(key_w, shape=(out_features, in_features)),
        "b": np.zeros(out_features),
    }

    return params
