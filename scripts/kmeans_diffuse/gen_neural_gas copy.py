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
""" Analytic Diffusion Model """
"""---------------------"""

print("\nTesting analytic diffusion")

key = jax.random.key(model_config["model"]["seed"])
xs, labels = next(iter(dataloader))

# Test the functions related to analytic score diffusion

ks = state["variables"]["params"]["c"].copy()
t0 = 0.1
phi_T = jax.random.truncated_normal(key, -1.0, 1.0, ks.shape[-1])
alpha_bar_t = sbdr.analytic_score_diffusion.cosine_schedule(t0)

print(f"\tks shape: {ks.shape}")
print(f"\tphi_T shape: {phi_T.shape}")
print(f"\tt0 : {t0}")
print(f"\talpha_bar_t: {alpha_bar_t}")


key = jax.random.key(70)

phi_ts, aux = sbdr.analytic_score_diffusion.diffuse_topk(
    key,
    phi_T,
    ks,
    T_max=1.0,
    num_steps=100,
    topk=10,
)

print(f"\n\tphi_ts shape: {phi_ts.shape}") # 
sbdr.print_pytree_shapes(aux)


"""---------------------"""
""" Nicely plot the diffusion process """
"""---------------------"""

# Create a plotly figure with subplots and a single slider
# The slider slides over the time axis (i.e., first dimension in this case) of phi_ts
# the first subplot shows the time series of phi_ts[i], reshaped to (28,28) plotted as a heatmap, and is updated by the slider
# the second subplot contains a scatter plot of all the (scalar) values in the aux dictionary over time, and is not updated by the slider

fig = make_subplots(rows=1, cols=2, subplot_titles=["Time series of phi_ts", "Aux values over time"])
fig.update_layout(height=600, width=1000)

# Add heatmap to the left subplot
fig.add_trace(
    go.Heatmap(
        z=phi_ts[0].reshape(28, 28)[::-1],
        colorscale="plasma",
        showscale=False,
        colorbar=dict(
            title="Value",
            xanchor="left",
            x=0.05
        )
    ),
    row=1, col=1
)

# Add scatter plot to the right subplot
for i, (k, v) in enumerate(aux.items()):
    fig.add_trace(
        go.Scatter(
            x=np.arange(v.shape[0]),
            y=v,
            mode="lines",
            name=k,
            showlegend=True,
        ),
        row=1, col=2
    )

# Create slider with the steps of phi_ts (only in the first subplot)
steps = []
for i in range(phi_ts.shape[0]):
    step = dict(
        method="update",
        args=[{"z": phi_ts[i].reshape(28, 28)[::-1]}, {"title": f"Time step {i}"}],
        label=f"Time step {i}",
    )
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time step: "},
    pad={"t": 50},
    steps=steps,
)]


fig.update_layout(
    sliders=sliders
)

fig.show()