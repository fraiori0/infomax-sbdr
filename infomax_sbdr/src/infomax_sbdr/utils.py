import os
import json
import pickle
import jax.numpy as np
import optax


# convolution operator over one specified axis, useful for time convolution with arbitrary batch dimensions
def conv1d(x, w, axis: int, mode="valid"):
    return np.apply_along_axis(lambda x: np.convolve(x, w, mode=mode), axis, x)


def print_pytree_shapes(pytree, prefix=""):
    if isinstance(pytree, dict):
        for key, value in pytree.items():
            print(f"{prefix}{key}:")
            print_pytree_shapes(value, prefix + "\t")
    elif isinstance(pytree, (list, tuple)):
        for idx, value in enumerate(pytree):
            print(f"{prefix}[{idx}]:")
            print_pytree_shapes(value, prefix + "\t")
    else:
        print(f"{prefix}{pytree.shape}")


def save_model(params, history, opt_state, model_path, verbose=True):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(os.path.join(model_path, "params.pkl"), "wb") as f:
        pickle.dump(params, f)

    with open(os.path.join(model_path, "opt_state.pkl"), "wb") as f:
        pickle.dump(opt_state, f)

    if verbose:
        print(f"Model saved: {model_path}")


def load_model(model_path, verbose=True):
    with open(os.path.join(model_path, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    with open(os.path.join(model_path, "opt_state.pkl"), "rb") as f:
        opt_state = pickle.load(f)

    if verbose:
        print(f"Model loaded: {model_path}")

    return params, opt_state
