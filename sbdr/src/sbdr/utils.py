import os
import json
import pickle
import jax.numpy as np


# convolution operator over one specified axis, useful for time convolution with arbitrary batch dimensions
def conv1d(x, w, axis: int, mode="valid"):
    return np.apply_along_axis(lambda x: np.convolve(x, w, mode=mode), axis, x)


def save_model(params, info_dict, model_path, model_name, verbose=True):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(os.path.join(model_path, model_name + "_params.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(model_path, model_name + "_info.json"), "w") as f:
        json.dump(info_dict, f)
    if verbose:
        print(f"Model saved: {model_name}")
