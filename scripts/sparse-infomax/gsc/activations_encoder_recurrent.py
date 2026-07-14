import os
import argparse

default_model = "rec"
default_number = "1"
default_cuda = "1"
CHKP_STEP = 148

save_prefix = "clean"

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    os.pardir,
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
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

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
import json

import numpy as onp

import infomax_sbdr as sbdr

from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms as tv_transforms

import cv2
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)

# print available devices
print(f"Available devices: {jax.devices()}")

# torch.multiprocessing.set_start_method('spawn')

# print the shapes nicely
def get_shapes(nested_dict):
    return jax.tree_util.tree_map(lambda x: x.shape, nested_dict)

"""---------------------"""
""" Import model config """
"""---------------------"""
model_folder = os.path.join(
    base_folder,
    "resources",
    "models",
    "gsc",
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
    "gsc",
)

MAX_SAMPLES: int | None = None # 2050          # ← set to e.g. 500 for quick testing
CACHE_DIR:   str | None = "./cache"     # ← set to None to skip disk caching

transform = sbdr.SpecAugmentTransform(
    **model_config["dataset"]["transform"]["specaugment"]["kwargs"],
)

dataset_train = sbdr.GSCDataset(
    root        = data_folder,
    split       = "train",
    precompute  = True,
    augment     = None,
    max_samples = MAX_SAMPLES,
    cache_dir   = CACHE_DIR,
)

dataset_val = sbdr.GSCDataset(
    root        = data_folder,
    split       = "val",
    precompute  = True,
    augment     = None,
    max_samples = MAX_SAMPLES,
    cache_dir   = CACHE_DIR,
)

dataloader_train = sbdr.NumpyLoader(
    dataset_train,
    batch_size=model_config["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
    generator=torch.Generator().manual_seed(model_config["model"]["seed"]),
)

dataloader_val = sbdr.NumpyLoader(
    dataset_val,
    batch_size=model_config["validation"]["dataloader"]["batch_size"],
    shuffle=False,
    drop_last=True,
    generator=torch.Generator().manual_seed(model_config["model"]["seed"]),
)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"Train samples : {len(dataset_train):>7,}   batches: {len(dataloader_train):,}")
print(f"Val   samples : {len(dataset_val):>7,}   batches: {len(dataloader_val):,}")
# print(f"Spectrogram   : ({N_MELS} mels × {N_FRAMES} frames)")
print(f"Cache dir     : {CACHE_DIR}")


print("\nSample shapes")
xs, labels = next(iter(dataloader_train))
print(f"xs: {xs.shape}")
print(f"labels: {labels.shape}")

# # Compute maximum and minimum per each feature across whole dataset 
# xmin = np.inf
# xmax = -np.inf
# for xs, _ in tqdm(dataloader_train, desc="Computing dataset min and max"):
#     batch_min = xs.reshape((-1, xs.shape[-1])).min(axis=0)
#     batch_max = xs.reshape((-1, xs.shape[-1])).max(axis=0)
#     xmin = np.minimum(xmin, batch_min)
#     xmax = np.maximum(xmax, batch_max)

# print(f"Min: {xmin}")
# print(f"Max: {xmax}")
# print(np.maximum(np.abs(xmin), np.abs(xmax)))


"""---------------------"""
""" Init Network """
"""---------------------"""

print("\nInitializing model")

model_class = sbdr.config_rpl_module_dict[model_config["model"]["type"]]

model = model_class(
    **model_config["model"]["kwargs"],
)

# # # Initialize parameters
# Take some data
x_seq, labels = next(iter(dataloader_train))
# Generate key
key = jax.random.key(model_config["model"]["seed"])
# Generate random initial state
s0 = model.init_state_from_input(key, x_seq[..., 0, :])
# Init params
variables = model.init(key, s0, x_seq[..., 0, :])

print(f"\tDict of variables: \n\t{variables.keys()}")

pprint(get_shapes(variables))

o, aux = model.apply(
    variables,
    model.init_state_from_input(key, x_seq[..., 0, :]),
    x_seq,
    method=model.scan,
)

"""---------------------"""
""" Forward Pass """
"""---------------------"""

print("\nForward pass jitted")

def forward(variables, xs, key):
    # Here we assume an input sequence, not a single step
    # i.e., second-to-last axis is time
    # generate initial state
    s0 = model.init_state_from_input(key, xs[..., 0, :])
    out = model.apply(
        variables,
        s0,
        xs,
        method=model.scan,
    )
    return out

forward_jitted = jit(forward)

# test the forward pass
xs, labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
key = jax.random.key(model_config["model"]["seed"])

outs, aux = forward_jitted(variables, xs, key)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))
print(f"\tAux shapes:")
pprint(get_shapes(aux))

"""---------------------"""
""" Checkpointing """
"""---------------------"""


checkpoint_manager = orbax.checkpoint.CheckpointManager(
    directory=os.path.join(model_folder, "manual_select"),
    # checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
    options=orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=model_config["training"]["checkpoint"]["save_interval"],
        max_to_keep=model_config["training"]["checkpoint"]["max_to_keep"],
        step_format_fixed_length=5,
    ),
)

print("\nCheckpointing stuff")

# add batch stats to train state
optimizer = sbdr.config_optimizer_dict[model_config["training"]["optimizer"]["type"]]
optimizer = optimizer(**model_config["training"]["optimizer"]["kwargs"])

state = {
    "variables": variables,
    "opt_state": optimizer.init(variables["params"]),
    "step": CHKP_STEP,
}

# # save state
# checkpoint_manager.save(
#     0,
#     args=orbax.checkpoint.args.StandardSave(state),
# )

state = checkpoint_manager.restore(
    step=CHKP_STEP, args=orbax.checkpoint.args.StandardRestore(state)
)

print(f"\tLoaded checkpoint: {CHKP_STEP}")


# # Reset upper layers, the first layer should converge first
# for l_idx in range(1, len(model_config["model"]["kwargs"]["features"])):
#     state["variables"]["params"][f"layers_{l_idx}"]["conv"] = variables["params"][f"layers_{l_idx}"]["conv"]

# overwrite "variables" using the current state (may be composed of imported checkpoints with some variable reset)
variables = state["variables"]


"""---------------------"""
""" Encode the entire dataset and save as sequences of SBDR """
"""---------------------"""

save_path = os.path.join(model_folder, "activations")
os.makedirs(save_path, exist_ok=True)

print("\nEncoding data")

def fast_jax_to_nested(data):
    # 1. Bring to CPU and flatten the sequence dimensions: (B, T, F) -> (B*T, F)
    B, T, F = data.shape
    data_flat = onp.asarray(data).reshape(-1, F)
    
    # 2. Get all non-zero coordinates at once
    row_idxs, feat_idxs = onp.nonzero(data_flat)
    
    # 3. Count how many active features are in each individual (b, t) step.
    # bincount tells us exactly how to slice our feature array.
    counts = onp.bincount(row_idxs, minlength=B*T)
    
    # 4. Calculate the split points (cumulative sum of counts)
    split_indices = onp.cumsum(counts)[:-1]
    
    # 5. Split the flat feature array into sub-arrays
    split_arrays = onp.split(feat_idxs, split_indices)
    
    # By calling .tolist() on each sub-array, NumPy instantly converts 
    # the internal elements from np.int64 to native Python ints.
    flat_time_steps = [x.tolist() for x in split_arrays]
    
    # 6. Reshape back into the original (B, T) nested structure
    return [flat_time_steps[i * T : (i + 1) * T] for i in range(B)]

def collect_data(dataloader, seed=CHKP_STEP):
    
    encodings = {k: [] for k in ["z", "zf", "label"]}
    encodings_sparse = {k: [] for k in ["z", "zf", "label"]}

    key = jax.random.key(seed)
    
    for batch_idx, (xs, labels) in tqdm(
        enumerate(dataloader), leave=False, total=len(dataloader)
    ):
        # quick debug
        if batch_idx > 10:
            break

        key, _ = jax.random.split(key)
        outs, aux = forward_jitted(variables, xs, key)
        
        # # # Sparsify
        # set to 1 values above a threshold
        thrsh = 0.15
        outs["z"] = np.where(outs["z"] > thrsh, 1.0, 0.0).astype(np.uint8)
        outs["zf"] = np.where(outs["zf"] > thrsh, 1.0, 0.0).astype(np.uint8)

        # append values
        encodings["z"].append(outs["z"])
        encodings["zf"].append(outs["zf"])
        encodings["label"].append(labels)

        # extend instead of append
        encodings_sparse["label"].extend(labels.astype(np.int32).tolist())
        # generate also, for each sample and for both z and zf,
        # a nested list with at each step a list of index of active units
        for k in ["z", "zf"]:
            # can we do this in parallel over time and batch dimensions?
            # i.e., without scanning over time or over batch dimensions
            # extend instead of append
            encodings_sparse[k].extend(fast_jax_to_nested(outs[k]))

    return encodings, encodings_sparse

train_e, train_e_sparse = collect_data(dataloader_train)

val_e, val_e_sparse = collect_data(dataloader_val)

print("\nEncoding done")

# concatenate and convert to uint8 z and zf
for k in train_e.keys():
    train_e[k] = np.concatenate(train_e[k], axis=0)
    val_e[k] = np.concatenate(val_e[k], axis=0)
    # convert to original numpy
    train_e[k] = onp.array(train_e[k])
    val_e[k] = onp.array(val_e[k])

print ("\nSaving data (NumPy)")
# Convert to original NumPy array and save compressed
onp.savez_compressed(
    os.path.join(model_folder, f"{save_prefix}_train.npz"),
    **train_e
)
onp.savez_compressed(
    os.path.join(model_folder, f"{save_prefix}_val.npz"),
    **val_e
)

print("\nSaving data (JSON)")
# Save also the sparse activation nested list
# as JSON file
with open(os.path.join(save_path, f"{save_prefix}_sparse_train.json"), "w") as f:
    json.dump(train_e_sparse, f)
with open(os.path.join(save_path, f"{save_prefix}_sparse_val.json"), "w") as f:
    json.dump(val_e_sparse, f)

print("\nSparsity")
# print quantile of per-sample sparsity and some statistics
qs = onp.array((0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
t_sample_act = train_e["z"].mean(-1).reshape(-1)
v_sample_act = val_e["z"].mean(-1).reshape(-1)
t_quantiles_samples = onp.quantile(t_sample_act, qs)
v_quantiles_samples = onp.quantile(v_sample_act, qs)

# print nicely
print(f"\t\ttrain\tval")
for i in range(len(qs)):
    print(f"{qs[i]:.2f}:\t{t_quantiles_samples[i]:.3f}\t{v_quantiles_samples[i]:.3f}")