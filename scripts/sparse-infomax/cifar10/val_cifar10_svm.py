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

BINARIZE = False  # whether to binarize the outputs or not
BINARIZE_THRESHOLD = None # threshold for binarization, only used if BINARIZE is True
BINARIZE_K = 15 # maximum number of non-zero elements to keep, if BINARIZE is True

# remember to change the pooling function in model definition, if using global pool model
default_model = "vgg_gavg_sigmoid_and" #"vgg_sigmoid_and"  # "vgg_sbdr_5softmax/1"  #
default_number = "2_bis"
default_checkpoint_subfolder = "manual_select" # 
default_step = 250  # 102

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
labels_categorical = labels_onehot.argmax(axis=-1)

print(f"\tEncoding shape (zs): {zs.shape}")
print(f"\tLabels shape (one-hot): {labels_onehot.shape}")
print(f"\tLabels shape (categorical): {labels_categorical.shape}")

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


labels_categorical_val = labels_onehot_val.argmax(axis=-1)

print(f"\tEncoding shape (zs_val): {zs_val.shape}")
print(f"\tLabels shape (one-hot): {labels_onehot_val.shape}")
print(f"\tLabels shape (categorical): {labels_categorical_val.shape}")

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
th_active = np.array((0.001, 0.01, 0.05, 0.1, 0.15))
count_activated = (zs > th_active[:, None, None]).sum(axis=(-2))
used_less_than = np.array((1.5, 2.5, 5.5, 10.5, 20.5))
count_unused = (count_activated[:, None] < used_less_than[None, :, None]).sum(axis=-1)
print(f"\nUnused units:")
for i, th in enumerate(th_active):
    print(f"\tThreshold {th:.3f}")
    print(
        f"\t  {count_unused[i]}"
    )

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
    zs = binarize_k_jitted(zs)
    zs_val = binarize_k_jitted(zs_val)


"""---------------------"""
""" K-Nearest Neighbors Classification """
"""---------------------"""

print("\nK-Nearest Neighbors Classification")

# # # Compute "distances" using the similarity metrics
sim_fn = jit(partial(
    sbdr.config_similarity_dict[model_config["validation"]["sim_fn"]["type"]],
    **model_config["validation"]["sim_fn"]["kwargs"],
))

ds_val = -sim_fn(zs_val[:, None], zs[None, :])

print(f"\tDistances shape: {ds_val.shape}")

# For each example in the validation set, find the k nearest neighbors in the training set
k = 19
# Use jax.lax.top_k to find the index of the k nearest neighbors
nearest_indices = jax.lax.top_k(-ds_val, k=k)[1]
# np.argpartition(ds_val, kth=k, axis=-1)[:, :k]
print(f"\tNearest indices shape: {nearest_indices.shape}")
# Select the labels of the k nearest neighbors
nearest_labels = labels_onehot[nearest_indices]
print(f"\tNearest labels shape: {nearest_labels.shape}")

# Check how many of the k nearest neighbors have the same label as the validation example
correct_mask = (labels_onehot_val[:, None] * nearest_labels).sum(axis=-1)  # shape: (n_val, k)
print(f"\tCorrect mask shape: {correct_mask.shape}")

# Compute the accuracy as the fraction of correct neighbors
acc_knn = correct_mask.mean()

correct_avg = correct_mask.mean(axis=-1)

print(f"\tKNN accuracy: {acc_knn}")
print(f"\tKNN accuracy per example: {correct_avg.mean()},  std: {correct_avg.std()}")

# Compute accuracy according to neighbor vote (with equal weights)
average_label = (nearest_labels.mean(axis=-2))
print(f"\tAverage label shape: {average_label.shape}")
# print(average_label[:5])
# Convert to categorical labels
labels_categorical_knn_val = average_label.argmax(axis=-1)
# Compute accuracy
acc_knn_vote = (labels_categorical_knn_val == labels_categorical_val).mean()

print(f"\tKNN accuracy (vote)")
print(f"\t\tK={k} : {acc_knn_vote:.4f}")


"""---------------------"""
""" Linear Support Vector Classification """
"""---------------------"""

print("\nLinear Support Vector Classification")



print(f"\tLabels train shape (categorical): {labels_categorical.shape}")
print(f"\tLabels val shape (categorical): {labels_categorical_val.shape}")


svm_model = LinearSVC(
    random_state=0,
    tol=1e-4,
    multi_class="ovr",
    intercept_scaling=1,
    C=5,
    penalty="l1",
    loss="squared_hinge",
)

print("\nTraining Linear SVM on the training set")
t0 = time()

svm_model.fit(
    zs,
    labels_categorical,
)
acc_train = svm_model.score(zs, labels_categorical)
acc_val = svm_model.score(zs_val, labels_categorical_val)

print(f"\t  Time: {time() - t0:.2f} seconds")

print(f"\tAccuracy on training set: {acc_train}")

print(f"\tAccuracy on validation set: {acc_val}")


# fit also a linear logistic regression for comparison
from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(
    random_state=0,
    tol=1e-4,
    multi_class="multinomial",
    C=1,
    penalty="l1",
    solver="saga",
)

print("\nTraining Linear Logistic Regression on the training set")
t0 = time()

logreg_model.fit(
    zs,
    labels_categorical,
)

acc_train_logreg = logreg_model.score(zs, labels_categorical)
acc_val_logreg = logreg_model.score(zs_val, labels_categorical_val)
print(f"\t  Time: {time() - t0:.2f} seconds")
print(f"\tAccuracy on training set: {acc_train_logreg}")
print(f"\tAccuracy on validation set: {acc_val_logreg}")
