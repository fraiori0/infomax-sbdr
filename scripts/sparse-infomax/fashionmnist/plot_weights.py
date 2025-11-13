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
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

# np.set_printoptions(precision=4, suppress=True)
pio.renderers.default = "browser"

BINARIZE = True  # whether to binarize the outputs or not
BINARIZE_THRESHOLD = None # threshold for binarization, only used if BINARIZE is True
BINARIZE_K = 15 # maximum number of non-zero elements to keep, if BINARIZE is True

# remember to change the pooling function in model definition, if using global pool model
default_model = "dense_sigmoid_logand" #"vgg_sigmoid_and"  # "vgg_sbdr_5softmax/1"  #
default_number = "5slab"
default_checkpoint_subfolder = "manual_select" # 
default_step = 45  # 102

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
    "fashionmnist",
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
    flatten=True
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
        # # BATCH_NORM - change here
        # mutable=["batch_stats"],
    )


forward_eval_jitted = jit(forward_eval)

# test the forward pass
xs, labels = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])

# # BATCH_NORM - change here
# outs, _ = forward_eval_jitted(variables, xs)
outs = forward_eval_jitted(variables, xs)

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
""" Plot the top-k weights activated by some sample input """
"""---------------------"""

print("\nPlot the top-k weights activated by some sample input")

N_TOP_K = 5

def keep_top_k(x, k):
    """ Keep only the k highest activations per sample, set rest to 0 """
    # x: (n_samples, n_features)
    # output: (n_samples, n_features)
    def keep_top_k_single(x_single, k):
        thresh = np.partition(x_single, -k)[-k]
        return np.where(x_single >= thresh, x_single, 0.0)
    # vmap over samples
    return jax.vmap(partial(keep_top_k_single, k=k))(x)

keep_top_k_jitted = jax.jit(partial(keep_top_k, k=N_TOP_K))

xs, labels = next(iter(dataloader_train))
xs, labels = next(iter(dataloader_train))

# Apply salt and pepper gaussian noise
p_binary = 0.1
std_noise = 0.4
key = jax.random.key(model_config["model"]["seed"])
key, _ = jax.random.split(key)
noise_mask = jax.random.bernoulli(key, p=0.1, shape=xs.shape)
key, _ = jax.random.split(key)
xs = xs + noise_mask * jax.random.normal(key, shape=xs.shape) * std_noise

# BATCH_NORM - change here
# outs, _ = forward_eval_jitted(variables, xs)
outs = forward_eval_jitted(variables, xs)

# Take the indices of the topk units
z = outs["z"]
z_topk_idx = np.argpartition(z, -N_TOP_K, axis=-1)[:, -N_TOP_K:]
print(f"\tShape of z: {z.shape}")
print(f"\tShape of z_topk_idx: {z_topk_idx.shape}")

# for each sample, select from the kernel of the first layer the weights corresponding to the top-k active units
kernel = variables["params"]["layers"]["layers_0"]["h"]["kernel"] # (in_features, out_features)
bias = variables["params"]["layers"]["layers_0"]["h"]["bias"] # (out_features,)

print("\tShape of kernel: ", kernel.shape)
print("\tShape of bias: ", bias.shape)

# Select the top k weights of topk units for each sample
kernel_topk = kernel[:, z_topk_idx] # (in_features, n_samples, n_topk)
kernel_topk = np.swapaxes(kernel_topk, 0, 1) # (n_samples, in_features, n_topk)
bias_topk = bias[z_topk_idx] # (n_samples, n_topk)

print("\tShape of kernel_topk: ", kernel_topk.shape)
print("\tShape of bias_topk: ", bias_topk.shape)

def topk_mask(z, kernel, bias):
    z_topk_idx = np.argpartition(z, -N_TOP_K, axis=-1)[..., -N_TOP_K:]
    kernel_topk = kernel[:, z_topk_idx] # (in_features, n_samples, n_topk)
    kernel_topk = np.moveaxis(kernel_topk, 0, -1) # (n_samples, n_topk, in_features)
    bias_topk = bias[z_topk_idx] # (n_samples, n_topk)

    return kernel_topk, bias_topk


# # plot input and weights for some units

# n_figs = 5

# n_plots = 1 + N_TOP_K
# n_cols = 3
# n_rows = n_plots // n_cols + (n_plots % n_cols > 0)

# for i in range(n_figs):

#     fig = make_subplots(
#         rows=n_rows, cols=n_cols,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.15,
#     )
    
#     # add the input image in the first place (1,1)
#     x = xs[i].reshape(28,28)
#     # flip y axis to correctly show image
#     x = np.flip(x, axis=0)
#     fig.add_trace(
#         go.Heatmap(
#             z=x,
#             colorscale="gray",
#             zmin=0.0,
#             zmax=1.0,
#             showscale=False,
#             xaxis="x1",
#             yaxis="y1",
#         ),
#         row=1, col=1,
#     )

#     # add input weights of N_TOP_K units
#     ws = kernel_topk[i]
#     bs = bias_topk[i]
#     for j in range(N_TOP_K):
#         w = ws[:, j].reshape(28,28)
#         # flip y axis to correctly show image
#         w = np.flip(w, axis=0)
#         fig.add_trace(
#             go.Heatmap(
#                 z=w,
#                 # colorscale="gray",
#                 # zmin=0.0,
#                 # zmax=1.0,
#                 showscale=False,
#                 xaxis="x1",
#                 yaxis="y1",
#             ),
#             row=(j+1)//n_cols+1, col=(j+1)%n_cols+1
#         )

#     # fig.update_layout(
#     #     width=800,
#     #     height=800,
#     #     showlegend=False,
#     # )

#     fig.update_xaxes(
#         visible=False,
#         showticklabels=False,
#     )

#     fig.update_yaxes(
#         visible=False,
#         showticklabels=False,
#     )

#     fig.show()

"""---------------------"""
""" Diffusion from kernel """
"""---------------------"""

print("\nDiffusion from kernel")

"""
Here we consider to use the input weights of active units as the null-space for a diffusion process.
We start from pure noise, and we apply perturbations.
A perturbations is pure noise from where we remove the components corresponding to the weights of the active units.
"""
@jit
def nullspace_projection(x, K):
    # Compute the null-space projector of the matrix K, assumed to have shape (n_components, n_features)
    # Formula : P = (I - K^T (K K^T)^{-1} K)
    # out = P x
    n = K.shape[-1]
    Q, _ = np.linalg.qr(K.T, mode='reduced')  # Q: (n, k)
    
    # P = I - Q @ Q^T
    I = np.eye(n)
    P = I - Q @ Q.T

    return P @ x

# Diffusion step
def diffusion_step(key, x, K, t):
    # t in [0, 1]
    # Compute noise, considering some noise schedule
    mu = x * np.sqrt(t)
    sigma_scale = 0.5
    sigma = sigma_scale*(1.0 - t)
    noise = jax.random.normal(key, mu.shape) * sigma

    # Apply noise
    x = mu + noise

    # Apply nullspace projection
    x = nullspace_projection(x, K)

    return x

def step(key, x, t):
    # Compute encoding
    outs = forward_eval_jitted(variables, x)
    z = outs["z"]
    # Select top-k kernel of the first layer
    kernel = variables["params"]["layers"]["layers_0"]["h"]["kernel"] # (in_features, out_features)
    bias = variables["params"]["layers"]["layers_0"]["h"]["bias"] # (out_features,)
    kernel_topk, bias_topk = topk_mask(z, kernel, bias)

    # Diffusion step
    x = diffusion_step(key, x, kernel_topk, t)

    return x

# test one step
key = jax.random.key(model_config["model"]["seed"])
xs, labels = next(iter(dataloader_train))

x0 = jax.random.normal(key, xs.shape[-1])*0.2

history = {
    "t": [],
    "x": [],
}
for t in np.linspace(0, 1, 50):
    history["t"].append(t)
    history["x"].append(x0.copy())
    key, _ = jax.random.split(key)
    x0 = step(key, x0, t)
    print(f"\tTime: {t:.2f} - Output shape: {x0.shape}")

# convert to numpy arrays
print("History shapes:")
for k in history.keys():
    history[k] = np.array(history[k])
    print(f"\t{k} : {history[k].shape}")

# Plot the evolution of the input

frames = []
for t in range(history["t"].shape[0]):
    frames.append(
        go.Frame(
            data=[go.Heatmap(
                z=history["x"][t].reshape(28, 28),
                colorscale='Viridis'
            )],
            name=str(t),  # Name for the frame
            layout=go.Layout(title_text=f"Time Step: {t}")
        )
    )
fig = go.Figure(
    data=[go.Heatmap(
        z=history["x"][0].reshape(28, 28),
        colorscale='Viridis',
    )],
    frames=frames
)


# Add slider
sliders = [dict(
    active=0,
    yanchor="top",
    y=-0.1,
    xanchor="left",
    currentvalue=dict(
        prefix="Time: ",
        visible=True,
        xanchor="right"
    ),
    pad=dict(b=10, t=50),
    len=0.9,
    x=0.1,
    steps=[
        dict(
            args=[[f.name], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
                transition=dict(duration=0)
            )],
            label=str(k),
            method="animate"
        )
        for k, f in enumerate(fig.frames)
    ]
)]


# Update layout
fig.update_layout(
    title="Heatmap Over Time",
    xaxis_title="Width",
    yaxis_title="Height",
    sliders=sliders,
    height=600,
    width=800
)

fig.show()



exit()

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
""" Save activations """
"""---------------------"""

# save the activations to a compressed npz file
save_folder = os.path.join(
    model_folder,
    "activations",
)
os.makedirs(save_folder, exist_ok=True)

onp.savez_compressed(
    os.path.join(save_folder, f"activations_chkp_{default_step:03d}.npz"),
    zs=onp.array(zs),
    labels_onehot=onp.array(labels_onehot),
    zs_val=onp.array(zs_val),
    labels_onehot_val=onp.array(labels_onehot_val),
)

exit()

