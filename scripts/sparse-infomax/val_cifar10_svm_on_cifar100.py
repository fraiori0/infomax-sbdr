import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "3"

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

from sklearn.svm import LinearSVC

# np.set_printoptions(precision=4, suppress=True)

BINARIZE = True  # whether to binarize the outputs or not
BINARIZE_THRESHOLD = None # threshold for binarization, only used if BINARIZE is True
BINARIZE_K = 7 # maximum number of non-zero elements to keep, if BINARIZE is True

default_model = "vgg_sigmoid_and" # "vgg_sigmoid_logand" #
default_number = "6" # "2" #
default_checkpoint_subfolder = "manual_select" # 
default_step = 265 # 295 #

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
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
    "cifar10",
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
    "cifar100",
)

print(f"\nLoading dataset from:\n\t{data_folder}")

# create the transform
transform = tv_transforms.Compose(
    [
        # normalize
        tv_transforms.Normalize(
            mean=model_config["dataset"]["transform"]["normalization"]["mean"],
            std=model_config["dataset"]["transform"]["normalization"]["std"],
        ),
        # change from  (C, H, W) to (H, W, C)
        tv_transforms.Lambda(lambda x: x.movedim(-3, -1)),
    ]
)

dataset = sbdr.Cifar100Dataset(
    folder_path=data_folder,
    kind="train",
    transform=transform,
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

print(f"\tExample batch")

xs, (fine_labels, coarse_labels) = next(iter(dataloader_train))

print(f"\t  Input xs: {xs.shape} (dtype: {xs.dtype})")
print(f"\t  Fine labels: {fine_labels.shape} (dtype: {fine_labels.dtype})")
print(f"\t  Coarse labels: {coarse_labels.shape} (dtype: {coarse_labels.dtype})")


"""---------------------"""
""" Init Model """
"""---------------------"""

print("\nInitializing model")

model_class = sbdr.config_module_dict[model_config["model"]["type"]]

model_eval = model_class(
    **model_config["model"]["kwargs"],
    activation_fn=sbdr.config_activation_dict[model_config["model"]["activation"]],
    out_activation_fn=sbdr.config_activation_dict[
        model_config["model"]["out_activation"]
    ],
    training=False,
)

# # # Initialize parameters
# Take some data
xs, (fine_labels, coarse_labels) = next(iter(dataloader_train))
# Generate key
key = jax.random.key(model_config["model"]["seed"])
# Init params and batch_stats
variables = model_eval.init(key, xs)

# # # Initialize the optimizer as well, to properly restore the full checkpoint
optimizer = sbdr.config_optimizer_dict[model_config["training"]["optimizer"]["type"]]
optimizer = optimizer(**model_config["training"]["optimizer"]["kwargs"])

state = {
    "variables": variables,
    "opt_state": optimizer.init(variables["params"]),
    "step": default_step,
}


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
""" Forward Pass """
"""---------------------"""

print("\nForward pass jitted")


def forward_eval(variables, xs):
    return model_eval.apply(
        variables,
        xs,
        # BATCH_NORM
        mutable=["batch_stats"],
    )


forward_eval_jitted = jit(forward_eval)

# test the forward pass
xs, (fine_labels, coarse_labels) = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])

outs, _ = forward_eval_jitted(variables, xs)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))

# print(f"\nTest time for one train epoch:")
# # test time for one epoch
# t0 = time()
# for xs, labels in tqdm(dataloader_train):
#     forward_eval_jitted(variables, xs)

# print(f"\tTime for one epoch: {time() - t0}")

"""---------------------"""
""" Forward pass on training set """
"""---------------------"""

print("\nForward pass on the whole training set")

# record output and labels
zs_train = []
fine_labels_train = []
coarse_labels_train = []

for xs, (fine_l, coarse_l) in tqdm(dataloader_train):
    # encode using a forward pass
    outs, _ = forward_eval_jitted(variables, xs)

    zs_train.append(outs["z"])
    fine_labels_train.append(fine_l)
    coarse_labels_train.append(coarse_l)

zs_train = np.concatenate(zs_train, axis=0)
fine_labels_train = np.concatenate(fine_labels_train, axis=0)
coarse_labels_train = np.concatenate(coarse_labels_train, axis=0)
fine_labels_train_cat = fine_labels_train.argmax(axis=-1)
coarse_labels_train_cat = coarse_labels_train.argmax(axis=-1)

print(f"\tEncoding shape (zs_train): {zs_train.shape}")
print(f"\tFine labels shape")
print(f"\t  one-hot: {fine_labels_train.shape}")
print(f"\t  categorical: {fine_labels_train_cat.shape}")
print(f"\tCoarse labels shape")
print(f"\t  one-hot: {coarse_labels_train.shape}")
print(f"\t  categorical: {coarse_labels_train_cat.shape}")


"""---------------------"""
""" Forward pass on validation set """
"""---------------------"""

print("\nForward pass on the whole validation set")

zs_val = []
fine_label_val = []
coarse_label_val = []

for xs, (fine_l, coarse_l) in tqdm(dataloader_val):
    # encode using a forward pass
    outs, _ = forward_eval_jitted(variables, xs)

    zs_val.append(outs["z"])
    fine_label_val.append(fine_l)
    coarse_label_val.append(coarse_l)

zs_val = np.concatenate(zs_val, axis=0)
fine_label_val = np.concatenate(fine_label_val, axis=0)
coarse_label_val = np.concatenate(coarse_label_val, axis=0)
fine_label_val_cat = fine_label_val.argmax(axis=-1)
coarse_label_val_cat = coarse_label_val.argmax(axis=-1)

print(f"\tEncoding shape (zs_val): {zs_val.shape}")
print(f"\tFine labels shape")
print(f"\t  one-hot: {fine_label_val.shape}")
print(f"\t  categorical: {fine_label_val_cat.shape}")
print(f"\tCoarse labels shape")
print(f"\t  one-hot: {coarse_label_val.shape}")
print(f"\t  categorical: {coarse_label_val_cat.shape}")


"""---------------------"""
""" Statistics on unit activity """
"""---------------------"""

print("\nStatistics on unit activity")

# Quantiles
qs = np.array((0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
per_unit_qs_train = np.quantile(zs_train.mean(axis=0), qs)
per_sample_qs_train = np.quantile(zs_train.mean(axis=-1), qs)

per_unit_qs_val = np.quantile(zs_val.mean(axis=0), qs)
per_sample_qs_val = np.quantile(zs_val.mean(axis=-1), qs)

print(f"\tQuantiles")
print(f"\t  qs = {qs}")
print(f"\t  Train")
print(f"\t    per unit: {per_unit_qs_train}")
print(f"\t    per sample: {per_sample_qs_train}")
print(f"\t  Val")
print(f"\t    per unit: {per_unit_qs_val}")
print(f"\t    per sample: {per_sample_qs_val}")


"""---------------------"""
""" Binarize/Sparsify encodings """
"""---------------------"""

if BINARIZE:
    print("\nBinarizing/Sparsifying the encodings")
    print(f"\tBinarization threshold: {BINARIZE_THRESHOLD}")
    print(f"\tMaximum number of non-zero elements: {BINARIZE_K}")

    def binarize_k(x, k):
        # keep only the k-top largest values
        idx_top_k = jax.lax.top_k(x, k)[1]
        # set the rest to zero, keep the original value only for the k-top largest values
        x_binarized = np.zeros_like(x).at[idx_top_k].set(x[idx_top_k])
        return x_binarized
    
    binarize_k_jitted = jit(vmap(partial(binarize_k, k=BINARIZE_K)))
    zs_train = binarize_k_jitted(zs_train)
    zs_val = binarize_k_jitted(zs_val)


"""---------------------"""
""" Linear Support Vector Classification """
"""---------------------"""

print("\nLinear Support Vector Classification")

svm_model_fine = LinearSVC(
    random_state=0,
    tol=1e-5,
    multi_class="ovr",
    dual=False,  # use primal solver, we have n_sample > n_features
    intercept_scaling = 10.0,
)

svm_model_coarse = LinearSVC(
    random_state=0,
    tol=1e-5,
    multi_class="ovr",
    dual=False,  # use primal solver, we have n_sample > n_features
    intercept_scaling = 10.0,
)




print("\tCoarse labels")

t0 = time()
svm_model_coarse.fit(onp.array(zs_train), onp.array(coarse_labels_train_cat))
print(f"\t  Time: {time() - t0:.2f} seconds")

# Use the model to make the prediction
coarse_l_pred_train_cat = svm_model_coarse.predict(onp.array(zs_train))
coarse_l_pred_val_cat = svm_model_coarse.predict(onp.array(zs_val))

# Compute accuracy
acc_coarse_train = (coarse_l_pred_train_cat == onp.array(coarse_labels_train_cat)).mean()
acc_coarse_val = (coarse_l_pred_val_cat == onp.array(coarse_label_val_cat)).mean()

print(f"\tAccuracy")
print(f"\t  Train: {acc_coarse_train}")
print(f"\t  Valid: {acc_coarse_val}")


print("\tFine labels")

t0 = time()
svm_model_fine.fit(onp.array(zs_train), onp.array(fine_labels_train_cat))
print(f"\t  Time: {time() - t0:.2f} seconds")

# Use the model to make the prediction
fine_l_pred_train_cat = svm_model_fine.predict(onp.array(zs_train))
fine_l_pred_val_cat = svm_model_fine.predict(onp.array(zs_val))

# Compute accuracy
acc_fine_train = (fine_l_pred_train_cat == onp.array(fine_labels_train_cat)).mean()
acc_fine_val = (fine_l_pred_val_cat == onp.array(fine_label_val_cat)).mean()


print(f"\tAccuracy")
print(f"\t  Train: {acc_fine_train}")
print(f"\t  Valid: {acc_fine_val}")

