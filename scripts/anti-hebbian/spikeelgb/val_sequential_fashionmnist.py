import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "0"

from functools import partial
import argparse
from time import time
from datetime import datetime
from typing import Callable, Any

import jax
from jax import vmap, grad, jit
import jax.numpy as np
import optax
import orbax.checkpoint
from flax.training import train_state
from pprint import pprint
import toml

import numpy as onp

import infomax_sbdr as sbdr

from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms as tv_transforms

import cv2
from tqdm import tqdm

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC, SVC

# np.set_printoptions(precision=4, suppress=True)

BINARIZE = True  # whether to binarize the outputs or not
BINARIZE_THRESHOLD = None # threshold for binarization, only used if BINARIZE is True
BINARIZE_K = 15 # maximum number of non-zero elements to keep, if BINARIZE is True

# remember to change the pooling function in model definition, if using global pool model
default_model = "td" #"vgg_sigmoid_and"  # "vgg_sbdr_5softmax/1"  #
default_number = "3"
default_checkpoint_subfolder = "manual_select" # 
default_step = 5  # 102

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    os.pardir,
)
base_folder = os.path.normpath(base_folder)


"""---------------------"""
""" Argument parsing """
"""---------------------"""

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="Name of model to train",
    default=default_model,
)
parser.add_argument(
    "-n",
    "--number",
    type=str,
    help="Number of model to train",
    default=default_number,
)

args = parser.parse_args()


"""---------------------"""
""" Import model config """
"""---------------------"""
model_folder = os.path.join(
    base_folder,
    "resources",
    "models",
    "antihebbian",
    args.model,
    args.number,
)

# import the elman_config.toml
with open(os.path.join(model_folder, "config.toml"), "r") as f:
    model_config = toml.load(f)

print(f"\nLoaded model config from:\n\t{model_folder}")
pprint(model_config)

"""---------------------"""
""" Dataloader and Transform """
"""---------------------"""

data_folder = os.path.join(
    base_folder,
    "resources",
    "datasets",
    "fashion-mnist",
    "data",
    "fashion",
)

dataset = sbdr.FashionMNISTPoissonDataset(
    folder_path=data_folder,
    kind="train",
    transform=None,
    flatten=True,
    **model_config["dataset"]["kwargs"],
)

dataset_train, dataset_val = torch.utils.data.random_split(
    dataset,
    [
        1 - model_config["validation"]["split"],
        model_config["validation"]["split"],
    ],
    generator=torch.Generator().manual_seed(model_config["model"]["seed"]),
)


dataloader_train = sbdr.NumpyLoader(
    dataset_train,
    batch_size=model_config["training"]["batch_size"],
    shuffle=False,
    drop_last=False,
)

dataloader_val = sbdr.NumpyLoader(
    dataset_val,
    batch_size=model_config["validation"]["dataloader"]["batch_size"],
    shuffle=False,
    drop_last=False,
)

xs, labels = next(iter(dataloader_train))

print(f"\tInput xs: {xs.shape} (dtype: {xs.dtype})")
print(f"\tLabels: {labels.shape} (dtype: {labels.dtype})")


"""---------------------"""
""" Visualize the images"""
"""---------------------"""


# def untransform(x):
#     return (x * np.array([0.2470, 0.2435, 0.2616])) + np.array([0.4914, 0.4822, 0.4465])


# # visualize the images
# # converted to BGR
# for i, (img_1, img_2) in enumerate(zip(xs_1, xs_2)):
#     img_1 = untransform(img_1)
#     img_2 = untransform(img_2)
#     # clip in range 0-1
#     img_1 = np.clip(img_1, 0, 1)
#     img_2 = np.clip(img_2, 0, 1)
#     # resize
#     img_1 = jax.image.resize(img_1, (128, 128, 3), "bilinear")
#     img_2 = jax.image.resize(img_2, (128, 128, 3), "bilinear")
#     # convert to BGR
#     img_1 = np.flip(img_1, axis=-1)
#     img_2 = np.flip(img_2, axis=-1)
#     # convert to 0-255
#     img_1 = (img_1 * 255).astype(np.uint8)
#     img_2 = (img_2 * 255).astype(np.uint8)
#     # convert to original NumPy array
#     img_1 = onp.array(img_1)
#     img_2 = onp.array(img_2)
#     # show image
#     cv2.imshow("Image 1", img_1)
#     cv2.imshow("Image 2", img_2)
#     k = cv2.waitKey(0)
#     if k == ord("q"):
#         break
# cv2.destroyAllWindows()

# exit()

"""---------------------"""
""" Init Network """
"""---------------------"""

print("\nInitializing model")

model_class = sbdr.config_ah_module_dict[model_config["model"]["type"]]

model_eval = model_class(
    **model_config["model"]["kwargs"],
)

# # # Initialize parameters
# Take some data
xs, labels = next(iter(dataloader_train))
# Generate a random initial state
key = jax.random.key(model_config["model"]["seed"])
s0 = model_eval.generate_initial_state(key, xs[..., 0, :])

# Init params and batch_stats
variables = model_eval.init(
    key,
    xs[..., 0, :],
    **s0,
)

print(f"\tDict of variables: \n\t{variables.keys()}")


# print the shapes nicely
def get_shapes(nested_dict):
    return jax.tree_util.tree_map(lambda x: x.shape, nested_dict)


pprint(get_shapes(variables))

"""---------------------"""
""" Forward Pass """
"""---------------------"""

print("\nForward scan jitted")


def forward_scan(variables, xs_seq, key):
    s0 = model_eval.generate_initial_state(key, xs_seq[..., 0, :])
    return model_eval.apply(
        variables,
        xs_seq,
        **s0,
        method=model_eval.forward_scan,
    )


forward_jitted = jit(forward_scan)

# test the forward scan
xs, labels = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])

outs = forward_jitted(variables, xs, key)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))

# print(f"\nTest one epoch:")
# # test time for one epoch
# t0 = time()
# for (xs_1, xs_2), labels in tqdm(dataloader_train):
#     key, _ = jax.random.split(key)
#     forward_jitted(variables, xs_1, key)

# print(f"\tTime for one epoch: {time() - t0}")


"""---------------------"""
""" Import checkpoint """
"""---------------------"""

print("\nImport checkpoint")

checkpoint_manager = orbax.checkpoint.CheckpointManager(
    directory=os.path.join(model_folder, default_checkpoint_subfolder),
    # checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
    options=orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=model_config["training"]["checkpoint"]["save_interval"],
        max_to_keep=model_config["training"]["checkpoint"]["max_to_keep"],
        step_format_fixed_length=5,
    ),
)

state = {
    "variables": variables,
    "step": 0,
}

state = checkpoint_manager.restore(
    step=default_step, args=orbax.checkpoint.args.StandardRestore(state)
)

print(f"\tDict of variables: \n\t{state['variables'].keys()}")

variables = state["variables"]


# print the shapes nicely
def get_shapes(nested_dict):
    return jax.tree_util.tree_map(lambda x: x.shape, nested_dict)


pprint(get_shapes(variables))


"""---------------------"""
""" Forward pass on training set """
"""---------------------"""

print("\nForward pass on the whole training set")

key = jax.random.key(model_config["model"]["seed"])

zs = []
labels_categorical = []

for i, (xs, labels) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):

    # # For quicker debugging
    # if i > 20:
    #     break

    # generate initial state
    key, _ = jax.random.split(key)
    # Compute outputs
    outs = forward_jitted(variables, xs, key)

    # Record temporal average of output
    zs.append(onp.array(outs["z"].mean(axis=-2)))

    # And categorical labels
    lab_cat = labels.argmax(axis=-1)
    labels_categorical.append(lab_cat)
    
# Convert to have numpy arrays

zs = onp.concatenate(zs, axis=0)
labels_categorical = onp.concatenate(labels_categorical, axis=0)

history = {
    "zs": zs,
    "labels_categorical": labels_categorical,
}

print(f"\tOutput shapes:")

pprint(get_shapes(history))

print(f"\tAverage Activity: {history['zs'].mean()}")


"""---------------------"""
""" Forward pass on validation set """
"""---------------------"""

print("\nForward pass on the whole validation set")

key = jax.random.key(model_config["model"]["seed"])

zs_val = []
labels_categorical_val = []

for i, (xs, labels) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):

    # # For quicker debugging
    # if i > 20:
    #     break

    # generate initial state
    key, _ = jax.random.split(key)
    # Compute outputs
    outs = forward_jitted(variables, xs, key)

    # Record temporal average of output
    zs_val.append(onp.array(outs["z"].mean(axis=-2)))

    # And categorical labels
    lab_cat = labels.argmax(axis=-1)
    labels_categorical_val.append(lab_cat)
    
# Convert to have numpy arrays

zs_val = onp.concatenate(zs_val, axis=0)
labels_categorical_val = onp.concatenate(labels_categorical_val, axis=0)

history_val = {
    "zs_val": zs_val,
    "labels_categorical_val": labels_categorical_val,
}

print(f"\tOutput shapes:")

pprint(get_shapes(history_val))

print(f"\tAverage Activity: {history_val['zs_val'].mean()}")


"""---------------------"""
""" Save outputs """
"""---------------------"""

# save the activations to a compressed npz file
save_folder = os.path.join(
    model_folder,
    "activations",
)
os.makedirs(save_folder, exist_ok=True)

onp.savez_compressed(
    os.path.join(save_folder, f"activations_chkp_{default_step:03d}.npz"),
    **history,
    **history_val,
)