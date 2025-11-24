import os
import argparse



default_model = "neural_gas"
default_number = "1"
default_cuda = "2"

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    # os.pardir,
)
base_folder = os.path.normpath(base_folder)

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

parser.add_argument(
    "-c",
    "--cuda_devices",
    type=str,
    help="Cuda available devices (as string)",
    default=default_cuda,
)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] =  args.cuda_devices


"""---------------------"""
""" Import libraries """
"""---------------------"""

from functools import partial

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
from plotly.subplots import make_subplots

np.set_printoptions(precision=4, suppress=True)

# print available devices
print(f"Available devices: {jax.devices()}")

# torch.multiprocessing.set_start_method('spawn')

"""---------------------"""
""" Import model config """
"""---------------------"""
model_folder = os.path.join(
    base_folder,
    "resources",
    "models",
    "kmeans_diffuse",
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

# create the transform
transform = tv_transforms.Compose(
    [
        # # random resize and cropping
        # tv_transforms.RandomResizedCrop(
        #     size=model_config["dataset"]["transform"]["resized_crop"]["size"],
        #     scale=model_config["dataset"]["transform"]["resized_crop"]["scale"],
        #     ratio=model_config["dataset"]["transform"]["resized_crop"]["ratio"],
        # ),
        # # random horizontal flip
        # tv_transforms.RandomHorizontalFlip(
        #     p=model_config["dataset"]["transform"]["flip"]["p"],
        # ),
        # # random color jitter
        # tv_transforms.ColorJitter(
        #     brightness=model_config["dataset"]["transform"]["color_jitter"][
        #         "brightness"
        #     ],
        #     contrast=model_config["dataset"]["transform"]["color_jitter"]["contrast"],
        #     saturation=model_config["dataset"]["transform"]["color_jitter"][
        #         "saturation"
        #     ],
        #     hue=model_config["dataset"]["transform"]["color_jitter"]["hue"],
        # ),
        # normalize
        tv_transforms.Normalize(
            mean=model_config["dataset"]["transform"]["normalization"]["mean"],
            std=model_config["dataset"]["transform"]["normalization"]["std"],
        ),
        # change from  (C, H, W) to (H, W, C)
        tv_transforms.Lambda(lambda x: x.movedim(-3, -1)),
    ]
)

dataset = sbdr.FashionMNISTDataset(
    folder_path=data_folder,
    kind="train",
    transform=transform,
    flatten=True,
)
dataset_test = sbdr.FashionMNISTDataset(
    folder_path=data_folder,
    kind="test",
    transform=transform,
    flatten=True,
)


dataloader = sbdr.NumpyLoader(
    dataset,
    batch_size=model_config["training"]["batch_size"],
    shuffle=model_config["training"]["dataloader"]["shuffle"],
    drop_last=model_config["training"]["dataloader"]["drop_last"],
    # num_workers=1,
)

dataloader_test = sbdr.NumpyLoader(
    dataset_test,
    batch_size=model_config["validation"]["dataloader"]["batch_size"],
    shuffle=False,
    drop_last=True,
    # num_workers=2,
)


xs, labels = next(iter(dataloader))
# print original device
print(f"\tOriginal device: {xs.device}")
print(f"\tInput xs: {xs.shape} (dtype: {xs.dtype})")
print(f"\tLabels: {labels.shape} (dtype: {labels.dtype})")


"""---------------------"""
""" Init Model """
"""---------------------"""

print("\nInitializing model")

model_class = sbdr.config_centroid_modules_dict[model_config["model"]["type"]]

model = model_class(
    **model_config["model"]["kwargs"],
)

# # # Initialize parameters
# Take some data
xs, labels = next(iter(dataloader))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
# Generate key
key = jax.random.key(model_config["model"]["seed"])
# Init params and batch_stats
variables = model.init(key, xs)

print(f"\tDict of variables: \n\t{variables.keys()}")


# print the shapes nicely
def get_shapes(nested_dict):
    return jax.tree_util.tree_map(lambda x: x.shape, nested_dict)

pprint(get_shapes(variables))


"""---------------------"""
""" Forward Pass """
"""---------------------"""

print("\nForward pass jitted")


def forward(variables, xs, key=None):
    return model.apply(
        variables,
        xs,
    )


forward_jitted = jit(forward)

# test the forward pass
xs, labels = next(iter(dataloader))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
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

def update_params(variables, *args, **kwargs):
    return model.apply(
        variables,
        variables,
        *args,
        **kwargs,
        method=model.update_params,
    )
update_params_jitted = jit(update_params)

# test params update

xs, labels = next(iter(dataloader))
key = jax.random.key(model_config["model"]["seed"])
outs = forward_jitted(variables, xs, key)
t = 0.2

_, dparams = update_params_jitted(variables, x=xs, out=outs, t=t)

print(f"\tInput shape: {xs.shape}")
print(f"\tdParams shapes:")
pprint(get_shapes(dparams))

"""---------------------"""
""" Checkpointing """
"""---------------------"""


checkpoint_manager = orbax.checkpoint.CheckpointManager(
    directory=os.path.join(model_folder, "checkpoints"),
    # checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
    options=orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=model_config["training"]["checkpoint"]["save_interval"],
        max_to_keep=model_config["training"]["checkpoint"]["max_to_keep"],
        step_format_fixed_length=5,
    ),
)

print("\nCheckpointing stuff")

state = {
    "variables": variables,
    "step": 0,
}

# # save state
# checkpoint_manager.save(
#     0,
#     args=orbax.checkpoint.args.StandardSave(state),
# )

LAST_STEP = checkpoint_manager.latest_step()
if LAST_STEP is None:
    LAST_STEP = 0
else:
    state = checkpoint_manager.restore(
        step=LAST_STEP, args=orbax.checkpoint.args.StandardRestore(state)
    )

variables = state["variables"]

print(f"\tLast step: {LAST_STEP}")


"""---------------------"""
""" Encode all train data """
"""---------------------"""

print("\nEncoding all train data")

key = jax.random.key(model_config["model"]["seed"])
xs, labels = next(iter(dataloader))
outs = forward_jitted(variables, xs, key)

history_train = {k : [] for k in outs.keys()}
history_train["x"] = []
history_train["label_cat"] = []

for xs, labels in tqdm(dataloader, total=len(dataloader)):
    key, _ = jax.random.split(key)
    outs = forward_jitted(variables, xs, key)

    for k in outs.keys():
        history_train[k].append(outs[k])
    history_train["x"].append(xs)
    history_train["label_cat"].append(np.argmax(labels, axis=-1))

# convert to numpy arrays
print("  History shapes:")
for k in history_train.keys():
    history_train[k] = np.concatenate(history_train[k], axis=0)
    print(f"\t{k} : {history_train[k].shape}")



"""---------------------"""
""" Encode all test data """
"""---------------------"""

print("\nEncoding all test data")

key = jax.random.key(model_config["model"]["seed"])
xs, labels = next(iter(dataloader))
outs = forward_jitted(variables, xs, key)


history_test = {k : [] for k in outs.keys()}
history_test["x"] = []
history_test["label_cat"] = []

for xs, labels in tqdm(dataloader_test, total=len(dataloader_test)):
    key, _ = jax.random.split(key)
    outs = forward_jitted(variables, xs, key)

    for k in outs.keys():
        history_test[k].append(outs[k])
    history_test["x"].append(xs)
    history_test["label_cat"].append(np.argmax(labels, axis=-1))

# convert to numpy arrays
print("  History shapes:")
for k in history_test.keys():
    history_test[k] = np.concatenate(history_test[k], axis=0)
    print(f"\t{k} : {history_test[k].shape}")

"""---------------------"""
""" Plot utils """
"""---------------------"""

def point_to_img(x):
    # Reshape
    x = x.reshape((*x.shape[:-1], 28, 28))
    # Rescale
    mu = np.array(model_config["dataset"]["transform"]["normalization"]["mean"])
    sigma = np.array(model_config["dataset"]["transform"]["normalization"]["std"])
    x = (x * sigma) + mu
    # Flip vertical axis
    x = x[..., ::-1, :]#, :]
    # Convert to unit8 in range [0, 255]
    x = np.clip((x * 255), 0, 255).astype(np.uint8)
    # Return as original numpy array
    return onp.array(x)
    

"""---------------------"""
""" Plot activation stats and covariance matrix """
"""---------------------"""

print("\nPlot activation stats and covariance matrix")

if False:
    cov_test = np.cov(history_test["z"].T)
    avg_per_unit_test = history_test["z"].mean(axis=0)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Activation stats",
            "Covariance matrix",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=avg_per_unit_test,
            nbinsx=100,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=cov_test,
        ),
        row=1, col=2
    )

    fig.show()


"""---------------------"""
""" Plot the centroids activated by some data point """
"""---------------------"""

if False:

    print("\nPlot the centroids activated by some data point")

    xs, labels = next(iter(dataloader))
    key = jax.random.key(model_config["model"]["seed"])

    outs = forward_jitted(variables, xs, key)

    # For each sample, take the points corresponding to the k closest centroids

    Cs = variables["params"]["c"]
    idx_topk = outs["i_sort"][..., :model_config["model"]["kwargs"]["topk"]]

    cs_topk = Cs[idx_topk, :]

    print(f"\tTopk shape: {cs_topk.shape}")


    # Plot in a Figure
    n_plots = model_config["model"]["kwargs"]["topk"] + 1
    n_cols = 5
    n_rows = n_plots // n_cols + (n_plots % n_cols > 0)

    for i, (x, cs_x) in enumerate(zip(xs, cs_topk)):

        if i>=3:
            break

        x_img = point_to_img(x)
        cs_x_img = point_to_img(cs_x)

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=(
                ["Input"] +
                [f"Unit {i}" for i in range(model_config["model"]["kwargs"]["topk"])]
            ),
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
        )

        # Add the input in the first place
        fig.add_trace(
            go.Heatmap(
                z=x_img,
                colorscale="gray",
                zmin=0, zmax=255,
            ),
            row=1, col=1,
        )

        # Add the rest of the units
        for j, cs_x_j in enumerate(cs_x_img):
            fig.add_trace(
                go.Heatmap(
                    z=cs_x_j,
                    colorscale="gray",
                    zmin=0, zmax=255,
                ),
                row=(j+1) // n_cols + 1, col=(j+1) % n_cols + 1,
            )

        fig.update_layout(
            # width=800,
            # height=600,
            template="plotly_white",
        )
        fig.show()
        


"""---------------------"""
""" Train a linear regressor for reconstruction """
"""---------------------"""

# Use scikit learn to train a multi-output linear regressor model for reconstruction

from sklearn.linear_model import LinearRegression

print("\nTrain a linear regressor for reconstruction")

if False:
    lin_model = LinearRegression(n_jobs = 7)

    x_regr = onp.array(history_test["z"])
    y_regr = onp.array(history_test["x"])

    lin_model.fit(x_regr, y_regr)
    y_pred = lin_model.predict(x_regr)

    print(f"Reconstruction R2 score: {lin_model.score(x_regr, y_regr)}")

    # Plot some example of original input, reconstruction, and difference
    # cols with different samples
    n_cols = 5
    # rows with original, reconstruction, difference
    n_rows = 3
    n_plots = n_cols * n_rows
    offset_plot = 0

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    for i, (x_original, x_pred) in enumerate(zip(
        y_regr[offset_plot:offset_plot+n_plots],
        y_pred[offset_plot:offset_plot+n_plots]
    )):
        x_img = point_to_img(x_original)
        x_regr_img = point_to_img(x_pred)

        fig.add_trace(
            go.Heatmap(
                z=x_img,
                colorscale="gray",
                zmin=0, zmax=255,
                showscale=False,
            ),
            row=1, col=(i+1) % n_cols + 1,
        )

        fig.add_trace(
            go.Heatmap(
                z=x_regr_img,
                colorscale="gray",
                zmin=0, zmax=255,
                showscale=False,
            ),
            row=2, col=(i+1) % n_cols + 1,
        )

        fig.add_trace(
            go.Heatmap(
                z=np.abs(x_img.astype(onp.float32) - x_regr_img.astype(onp.float32)),
                colorscale="viridis",
                zmin=-255, zmax=255,
            ),
            row=3, col=(i+1) % n_cols + 1,
        )

    fig.update_layout(
        # width=800,
        # height=600,
        template="plotly_white",
    )
    fig.show()



"""---------------------"""
""" Train a linear classifier using logistic regression """
"""---------------------"""

from sklearn.linear_model import LogisticRegression

if True:

    lin_class_model = LogisticRegression(
        random_state=0,
        tol=1e-4,
        multi_class="multinomial",
        C=1,
        # penalty="l1",
        # solver="saga",
        n_jobs=7,
    )

    x_regr_train = onp.array(history_train["z"])
    y_regr_train = onp.array(history_train["label_cat"])
    x_regr_test = onp.array(history_test["z"])
    y_regr_test = onp.array(history_test["label_cat"])

    lin_class_model.fit(x_regr_train, y_regr_train)

    print(f"Classification accuracy:")
    print(f"\ttrain:  {lin_class_model.score(x_regr_train, y_regr_train)}")
    print(f"\ttest:   {lin_class_model.score(x_regr_test, y_regr_test)}")