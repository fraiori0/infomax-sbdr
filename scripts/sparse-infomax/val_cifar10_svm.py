import os
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

BINARIZE = False  # whether to binarize the outputs or not
BINARIZE_THRESHOLD = 0.2  # threshold for binarization, only used if BINARIZE is True

default_model = "vgg_sigmoid_and"  # "vgg_sbdr_5softmax/1"  #
default_number = "6"
default_checkpoint_subfolder = "manual_select" # 
default_step = 265  # 102

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
    "cifar10",
)

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

dataset = sbdr.Cifar10Dataset(
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
xs, labels = next(iter(dataloader_train))
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
xs, labels = next(iter(dataloader_train))
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

xs, labels = next(iter(dataloader_train))
outs, _ = forward_eval_jitted(variables, xs)
sem_labels = np.zeros((labels.shape[-1], outs["z"].shape[-1]))
label_count = np.zeros(labels.shape[-1])


# record output and labels
zs = []
labels_onehot = []

for xs, labels in tqdm(dataloader_train):
    # encode using a forward pass
    outs, _ = forward_eval_jitted(variables, xs)

    zs.append(outs["z"])
    labels_onehot.append(labels)

    label_masks = (labels > 0.5).T

    for i, l in enumerate(label_masks):
        # print(l)
        sem_labels = sem_labels.at[i].set(sem_labels[i] + outs["z"][l].sum(axis=0))
        label_count = label_count.at[i].set(label_count[i] + l.sum())

    # break

zs = np.concatenate(zs, axis=0)
labels_onehot = np.concatenate(labels_onehot, axis=0)

print(f"\tEncoding shape (zs): {zs.shape}")
print(f"\tLabels shape (one-hot): {labels_onehot.shape}")

sem_labels = sem_labels / label_count[:, None]

# print(sem_labels[0])
# print(sem_labels[:, :10])
print(f"\nSemantic Labels")
print(f"\tPer-label count: {label_count}")
print(f"\tAverage per-label activity: {sem_labels.mean(axis=-1)}")
print(f"\tSemantic labels shape: {sem_labels.shape}")

print(f"\nAverage activity: {outs['z'].mean()}")

"""---------------------"""
""" Forward pass on validation set """
"""---------------------"""

print("\nForward pass on the whole validation set")

zs_val = []
labels_onehot_val = []

for xs, labels in tqdm(dataloader_val):
    # encode using a forward pass
    outs, _ = forward_eval_jitted(variables, xs)

    zs_val.append(outs["z"])
    labels_onehot_val.append(labels)

zs_val = np.concatenate(zs_val, axis=0)
labels_onehot_val = np.concatenate(labels_onehot_val, axis=0)

print(f"\tEncoding shape (zs_val): {zs_val.shape}")
print(f"\tLabels shape (one-hot): {labels_onehot_val.shape}")

"""---------------------"""
""" Statistics on unit activity """
"""---------------------"""

print("\nStatistics on unit activity")

# Quantiles
qs = np.array((0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
per_unit_qs = np.quantile(zs.mean(axis=0), qs)
per_sample_qs = np.quantile(zs.mean(axis=-1), qs)

print(f"\tPer-unit quantiles: {per_unit_qs}")
print(f"\tPer-sample quantiles: {per_sample_qs}")

# # # Histograms
print(f"\nHistograms")
n_bins = 20
# # Per-Unit Average Activity
bin_edges = np.geomspace(zs.mean(axis=0).min() + 1e-8, zs.mean(axis=0).max(), n_bins)
bin_edges = np.append(bin_edges, 1.5) - 1e-8
per_unit_hist, _ = np.histogram(zs.mean(axis=0), bins=bin_edges)
print(f"\tBin Count:\n\t\t{per_unit_hist}")
print(f"\tBin Centers:\n\t\t{((bin_edges[:-1] + bin_edges[1:]) / 2.0)}")

# # # Sharpness of unit activity
th_low = np.array((0.01, 0.05, 0.1, 0.15))
th_high = 1 - th_low
# Count the relative number of units below th_low and above th_high, and in the middle
count_low = (zs[None] < th_low[:, None, None]).mean(axis=(-1, -2))
count_middle = (
    (zs[None] > th_low[:, None, None]) & (zs[None] < th_high[:, None, None])
).mean(axis=(-1, -2))
count_high = (zs[None] > th_high[:, None, None]).mean(axis=(-1, -2))
print(f"\nSharpness of unit activity:")
print(f"\tLow: {count_low}")
print(f"\tMiddle: {count_middle}")
print(f"\tHigh: {count_high}")

# # # Unused units
# Count the relative number of units that are never active above some threshold
th_active = 0.5
count_activated = (zs > th_active).sum(axis=(0))
used_less_than = np.array((1.0, 2.0, 5.0, 10.0, 20.0))
count_unused = (count_activated[None, :] < used_less_than[:, None]).sum(axis=-1)
print(f"\nUnused units:")
print(f"\tUsed less than: {used_less_than}")
print(f"\tUnused: {count_unused}")


"""---------------------"""
""" Linear Support Vector Classification """
"""---------------------"""

print("\nLinear Support Vector Classification")

labels_categorical = labels_onehot.argmax(axis=-1)
labels_categoriacl_val = labels_onehot_val.argmax(axis=-1)

print(f"\tLabels train shape (categorical): {labels_categorical.shape}")
print(f"\tLabels val shape (categorical): {labels_categoriacl_val.shape}")


svm_model = LinearSVC(
    random_state=0,
    tol=1e-5,
    multi_class="ovr",
    dual=False,  # use primal solver, we have n_sample > n_features
)

print("\nTraining Linear SVM on the training set")
t0 = time()

if BINARIZE:
    svm_model.fit(
        (zs>0.2).astype(zs),
        labels_categorical,
    )
    acc_train = svm_model.score((zs>BINARIZE_THRESHOLD).astype(zs), labels_categorical)
    acc_val = svm_model.score((zs_val>BINARIZE_THRESHOLD).astype(zs_val), labels_categoriacl_val)
else:
    svm_model.fit(
        zs,
        labels_categorical,
    )
    acc_train = svm_model.score(zs, labels_categorical)
    acc_val = svm_model.score(zs_val, labels_categoriacl_val)

print(f"\t  Time: {time() - t0:.2f} seconds")

print(f"\tAccuracy on training set: {acc_train}")

print(f"\tAccuracy on validation set: {acc_val}")


exit()

"""---------------------"""
""" Inter-Label similarity """
"""---------------------"""

sim_fn = partial(
    sbdr.config_similarity_dict[model_config["validation"]["sim_fn"]["type"]],
    **model_config["validation"]["sim_fn"]["kwargs"],
)


# # # Similarity matrix between semantic labels (normalizing and using dot-product)
sem_labels = sem_labels / np.linalg.norm(sem_labels, axis=-1, keepdims=True)
label_sims = (sem_labels[:, None] * sem_labels[None, :]).sum(axis=-1)

# # # Similarity matrix between semantic labels (setting 90th percentile to 1 and using jaccard index)
# for each label, compute the 90th percentile of activity (in that label)
# q = model_config["validation"]["sim_fn"]["quantile"]
# sem_thresh = np.quantile(sem_labels, q, axis=-1)
# sem_labels = (sem_labels > sem_thresh[:, None]).astype(np.float32)
# # keep only the features shared with less then 2.5 labels
# print(sem_labels.mean(axis=-1))
# sem_labels = sem_labels * (sem_labels.sum(axis=0, keepdims=True) < 2.5)


print(sem_labels.shape)
print(sem_labels.mean(axis=-1))


label_sims = sim_fn(sem_labels[:, None], sem_labels[None, :])

print(f"\tSemantic label similarity: {label_sims}")
print(label_sims.shape)

# # plot on a heatmap
# fig = go.Figure()
# fig.add_trace(
#     go.Heatmap(
#         z=label_sims,
#         # zmin=0.0,
#         # zmax=1.0,
#         colorscale="Viridis",
#     )
# )
# fig.update_layout(
#     title="Semantic Label Similarity",
#     xaxis_title="Semantic Label",
#     yaxis_title="Semantic Label",
# )
# fig.show()


"""---------------------"""
""" Classification """
"""---------------------"""

print("\nClassification of validation set")

acc_top_1 = []
acc_top_3 = []

key = jax.random.key(model_config["model"]["seed"])
for xs, labels in tqdm(dataloader_val):

    # # add some gaussian noise
    # key, _ = jax.random.split(key)
    # noise = 0.2 * jax.random.normal(key, shape=xs.shape)
    # xs = xs + noise

    # encode using a forward pass
    outs, _ = forward_eval_jitted(variables, xs)

    # compute similarity to each semantic label
    # label_sims = sim_fn(outs["z"][:, None], sem_labels[None, :])
    label_sims = (outs["z"][:, None] * sem_labels[None, :]).sum(axis=-1)

    # Top 1 accuracy
    # find the most similar semantic label for each example
    label_ids = label_sims.argmax(axis=-1)
    # compute accuracy
    acc_top_1.append((label_ids == labels.argmax(axis=-1)).mean())

    # Top 3 accuracy
    # find the top 3 most similar semantic labels for each example
    label_ids = label_sims.argsort(axis=-1)[:, -3:]
    # compute accuracy
    acc_top_3.append(
        ((label_ids == labels.argmax(axis=-1)[:, None]).sum(axis=-1)).mean()
    )

    # break

# convert to jax numpy array
acc_top_1 = np.array(acc_top_1)
acc_top_3 = np.array(acc_top_3)

print(f"\tTop 1 accuracy: {acc_top_1.mean()}")
print(f"\tTop 3 accuracy: {acc_top_3.mean()}")
