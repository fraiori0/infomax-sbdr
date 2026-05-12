import os
import argparse

default_model = "tcn"
default_number = "3poolg"
default_cuda = "1"

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

model_eval = model_class(
    **model_config["model"]["kwargs"],
)

# # # Initialize parameters
# Take some data
x_seq, labels = next(iter(dataloader_train))
# Generate key
key = jax.random.key(model_config["model"]["seed"])
# Init params
variables = model.init(key, x_seq)

print(f"\tDict of variables: \n\t{variables.keys()}")

pprint(get_shapes(variables))

"""---------------------"""
""" Forward Pass """
"""---------------------"""

print("\nForward pass jitted")

def augment_input(xs, key):
    # roll on the time axis by a random number of steps
    # between 0 and N_FRAMES
    MAX_STEPS = 5
    n_steps = jax.random.randint(key, minval=-MAX_STEPS, maxval=MAX_STEPS, shape=())
    return np.roll(xs, n_steps, -2)


def forward(variables, xs, key):
    xs = augment_input(xs, key)
    out = model.apply(
        variables,
        xs,
    )
    return out


def forward_eval(variables, xs, key):
    out = model_eval.apply(
        variables,
        xs,
    )
    return out


forward_jitted = jit(forward)
forward_eval_jitted = jit(forward_eval)

# test the forward pass
xs, labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
key = jax.random.key(model_config["model"]["seed"])

# BATCH_NORM - change here
# outs, mutable_updates = forward_jitted(variables, xs_1, key)
# _, _ = forward_eval_jitted(variables, xs_1)
outs, aux = forward_jitted(variables, xs, key)
_, _ = forward_eval_jitted(variables, xs, key)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))
print(f"\tAux shapes:")
pprint(get_shapes(aux))

# BATCH_NORM - change here
# print(f"\tMutable updates:")
# pprint(get_shapes(mutable_updates))

# print(f"\nTest one epoch:")
# # test time for one epoch
# t0 = time()
# for (xs_1, xs_2), labels in tqdm(dataloader_train):
#     key, _ = jax.random.split(key)
#     forward_jitted(variables, xs_1, key)

# print(f"\tTime for one epoch: {time() - t0}")

# for k in outs[0].keys():
#     print(f"\n{k}")
#     print(f"\tmin: {outs[0][k].min()}")
#     print(f"\tmax: {outs[0][k].max()}")
#     print(f"\tmean: {outs[0][k].mean()}")
#     print(f"\tstd: {outs[0][k].std()}")

"""---------------------"""
""" Loss Function """
"""---------------------"""

def w_infonce(w):
    # Compute an InfoNCE loss with custom critic, given a weight matrix 
    # the weight matrix is assumed to be of shape (in_features, out_features)
    # and InfoNCE is computed to separate weights across units

    eps = model_config["training"]["loss"]["w_eps"]

    w = w.T # reshape to (out_features, in_features)
    w_avg = w.mean(0)

    p_ii = (w*w).sum(-1) + eps
    p_avg = (w*w_avg).sum(-1) + eps
    w_loss = -np.log(p_ii / p_avg).mean()

    aux = {
        "norm" : np.linalg.norm(w, axis=-1).mean()
    }

    return w_loss, aux


def w_abs_infonce(w):

    eps = model_config["training"]["loss"]["w_eps"]

    # take absolute value to avoid issues with logarithms
    w = np.abs(w)

    # Reshape to (out_features, in_features), we want to contrast the weights across units
    w = w.T
    # Compute average for contrasting
    w_avg = w.mean(0)

    # Compute InfoNCE loss
    p_ii = (w*w).sum(-1) + eps
    p_avg = (w*w_avg).sum(-1) + eps
    w_loss = -np.log(p_ii / p_avg).mean()

    aux = {
        "norm" : np.linalg.norm(w, axis=-1).mean()
    }

    return w_loss, aux

def encoder_infonce(a, eps=None):
    # given a vector of non-negative values
    # and shape (*batch_dims, time, features)
    if eps is None:
        eps = model_config["training"]["loss"]["eps"]

    # Compute average activation (for contrasting)
    a_avg = a.reshape((-1, a.shape[-1])).mean(0)
    
    # Compute InfoNCE loss to separate activations across al samples
    p_ii = (a * a).sum(-1) + eps
    p_avg = (a * a_avg).sum(-1) + eps
    loss_val = -np.log(p_ii / p_avg).mean()

    aux = {
        "infonce" : loss_val,
    }

    return loss_val, aux

def time_infonce(a, eps=None):
    # given a vector of non-negative values
    # and shape (*batch_dims, time, features)
    if eps is None:
        eps = model_config["training"]["loss"]["eps"]

    # Here we contrast the sequence of activation of different units
    # such that units should be active at different times
    a = np.swapaxes(a, -1, -2)
    a_avg = a.reshape((-1, a.shape[-1])).mean(0)
    
    # Compute InfoNCE loss to separate activations across al samples
    p_ii = (a * a).sum(-1) + eps
    p_avg = (a * a_avg).sum(-1) + eps
    loss_val = -np.log(p_ii / p_avg).mean()

    aux = {
        "infonce" : loss_val,
    }

    return loss_val, aux

def classification_loss(logits, labels):
    # Compute one-hot labels
    labels_onehot = jax.nn.one_hot(labels, logits.shape[-1])
    # Average logits over time axis
    # # taking only n_last steps
    # n_last = model_config["training"]["loss"]["class_steps"]
    # logits = logits[..., -n_last:, :].sum(-2)
    # Sum logits, such that the contribution of uninformative steps
    # can be more easily reduced
    logits = logits.sum(-2)
    # Compute categorical crossentropy loss
    loss = -(labels_onehot * jax.nn.log_softmax(logits)).sum(axis=-1)
    # compute accuracy
    aux = {
        "acc": (labels[:, None] == logits.argmax(-1)).mean(),
    }
    return loss.mean(), aux

# test loss function
xs, labels = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])
outs, o_aux = forward(variables, xs, key)


print("\nTesting loss function")
for l_idx in range(len(outs)):
    z = outs[l_idx]["z"]
    e_loss_val, e_aux = encoder_infonce(z)
    t_loss_val, t_aux = time_infonce(z)
    print(f"\tLosses: layer {l_idx}: e: {e_loss_val}, t: {t_loss_val}")

class_loss_val, class_aux = classification_loss(o_aux["logit"], labels)
print(f"\tLosses: class: {class_loss_val} (accuracy: {class_aux['acc']})")


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

# add batch stats to train state
optimizer = sbdr.config_optimizer_dict[model_config["training"]["optimizer"]["type"]]
optimizer = optimizer(**model_config["training"]["optimizer"]["kwargs"])

state = {
    "variables": variables,
    "opt_state": optimizer.init(variables["params"]),
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

print(f"\tLast step: {LAST_STEP}")


# # Reset upper layers, the first layer should converge first
# for l_idx in range(1, len(model_config["model"]["kwargs"]["features"])):
#     state["variables"]["params"][f"layers_{l_idx}"]["conv"] = variables["params"][f"layers_{l_idx}"]["conv"]

# overwrite "variables" using the current state (may be composed of imported checkpoints with some variable reset)
variables = state["variables"]


"""---------------------"""
""" Utils """
"""---------------------"""

Q_KEYS = ["0.05", "0.25", "0.5", "0.75", "0.95"]
QS = np.array((0.05, 0.25, 0.5, 0.75, 0.95))

@jit
def compute_metrics(outs):
    """Compute metrics from model outputs."""
    metrics = {}

    for l_idx, l_out in enumerate(outs):
        for k in ["y", "z", "p"]:
            
            # per-unit average
            unit_avg = np.mean(l_out[k].reshape((-1, l_out[k].shape[-1])), axis=0)
            qs_unit_val = np.quantile(unit_avg, QS)

            # per-sample average
            sample_avg = np.mean(l_out[k].reshape((-1, l_out[k].shape[-1])), axis=-1)
            qs_sample_val = np.quantile(sample_avg, QS)


            metrics.update(
                {f"unit_{l_idx}_{k}/{kk}": v for kk, v in zip(Q_KEYS, qs_unit_val)}
            )
            metrics.update(
                {f"sample_{l_idx}_{k}/{kk}": v for kk, v in zip(Q_KEYS, qs_sample_val)}
            )

    return metrics



"""---------------------"""
""" Training and Evaluation Steps """
"""---------------------"""

print("\nTraining and Evaluation Steps")


# @jit
def train_step(state, batch):
    
    def loss_fn(params):
        # Apply the model
        outs, o_aux = forward(
            {
                "params": params,
            },
            batch["x_1"],
            batch["key_1"],
        )

        # compute loss for each layer
        loss_val = 0.0
        metrics = compute_metrics(outs)
        for l_idx in range(len(outs)):

            z = outs[l_idx]["z"]
            p = outs[l_idx]["p"]
            # w = params[f"layers_{l_idx}"]["conv"]["kernel"]

            # p_loss_val, p_aux = pred_infonce(z, labels)
            z_loss_val, z_aux = encoder_infonce(z)
            t_loss_val, t_aux = time_infonce(z)
            # w_loss_val, w_aux = w_abs_infonce(w)

            layer_loss_val = (
                # 0.1 * w_loss_val
                + z_loss_val
                + 0*t_loss_val
            )

            loss_val = loss_val + layer_loss_val

            metrics.update({f"loss/{l_idx}": layer_loss_val})
            for k, v in z_aux.items():
                metrics.update({f"aux_losses_{l_idx}/z_{k}": v})
            # for k, v in w_aux.items():
            #     metrics.update({f"aux_losses_{l_idx}/w_{k}": v})
            for k, v in t_aux.items():
                metrics.update({f"aux_losses_{l_idx}/t_{k}": v})
            
            # # Train only first layer
            # break

        # # # Compute classification loss
        labels = batch["labels"]
        logits = o_aux["logit"]
        class_loss_val, class_aux = classification_loss(logits, labels)
        # Update loss_val
        loss_val = loss_val + 3*class_loss_val

        # Update metrics with classification
        metrics.update({"class/loss": class_loss_val})
        for k, v in class_aux.items():
            metrics.update({"class/" + k: v})

        others = {
            # BATCH_NORM - change here
            # "mutable_updates": jax.tree.map(
            #     lambda x, y: (x + y) / 2.0, mutable_updates_1, mutable_updates_2
            # ),
        }

        return loss_val, (metrics, others)

    # compute gradient, loss, and aux
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss_val, (metrics, others)), grads = grad_fn(state["variables"]["params"])
    # update weights
    updates, opt_state = optimizer.update(grads, state["opt_state"], state["variables"]["params"])
    new_params = optax.apply_updates(
        state["variables"]["params"], updates
    )
    # Update optimizer state
    state["opt_state"] = opt_state

    # Set new params
    state["variables"]["params"] = new_params

    # update step
    state["step"] += 1

    return state, metrics, grads


@jit
def eval_step(state, batch):
    
    def loss_fn(params):
        # Apply the model
        outs, o_aux = forward_eval(
            {
                "params": params,
            },
            batch["x_1"],
            batch["key_1"],
        )

        # compute loss for each layer
        loss_val = 0.0
        metrics = compute_metrics(outs)
        for l_idx in range(len(outs)):

            z = outs[l_idx]["z"]
            p = outs[l_idx]["p"]
            # w = params[f"layers_{l_idx}"]["conv"]["kernel"]

            # p_loss_val, p_aux = pred_infonce(z, labels)
            z_loss_val, z_aux = encoder_infonce(z)
            t_loss_val, t_aux = time_infonce(z)
            # w_loss_val, w_aux = w_abs_infonce(w)

            layer_loss_val = (
                # 0.1 * w_loss_val
                + z_loss_val
                + t_loss_val
            )

            loss_val = loss_val + layer_loss_val

            metrics.update({f"loss/{l_idx}": layer_loss_val})
            for k, v in z_aux.items():
                metrics.update({f"aux_losses_{l_idx}/z_{k}": v})
            # for k, v in w_aux.items():
            #     metrics.update({f"aux_losses_{l_idx}/w_{k}": v})
            for k, v in t_aux.items():
                metrics.update({f"aux_losses_{l_idx}/t_{k}": v})
            
            # # Train only first layer
            # break

        # # # Compute classification loss
        labels = batch["labels"]
        logits = o_aux["logit"]
        class_loss_val, class_aux = classification_loss(logits, labels)
        # Update loss_val
        loss_val = loss_val + 2*class_loss_val

        # Update metrics with classification
        metrics.update({"class/loss": class_loss_val})
        for k, v in class_aux.items():
            metrics.update({"class/" + k: v})

        # Used when considering batch norm mutables
        others = {
            # BATCH_NORM - change here
            # "mutable_updates": jax.tree.map(
            #     lambda x, y: (x + y) / 2.0, mutable_updates_1, mutable_updates_2
            # ),
        }

        return loss_val, (metrics, others)

    # compute loss
    loss_val, (metrics, others) = loss_fn(state["variables"]["params"])

    return metrics


print("\t Test one train step and one eval step")

xs, labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
key = jax.random.key(model_config["model"]["seed"])
key_1, key_2 = jax.random.split(key)
batch = {
    "x_1": xs,
    "key_1": key_1,
    "x_2": xs,
    "key_2": key_2,
    "labels": labels,
}

state, metrics, grads = train_step(state, batch)
metrics_val = eval_step(state, batch)

print("\nMetrics")
pprint(metrics)

print("\nGradient checks:")
gradients_are_finite = jax.tree_util.tree_map(
    lambda x: np.all(np.isfinite(x)).item(), grads
)
gradients_are_zero = jax.tree_util.tree_map(
    lambda x: np.all(np.isclose(x, 0.0, rtol=1e-20, atol=1e-20)).item(), grads
)
gradients_are_nan = jax.tree_util.tree_map(lambda x: np.any(np.isnan(x)).item(), grads)
print(f"\tGradients are FINITE:")
pprint(gradients_are_finite)
print(f"\tGradients are ZERO (tolerance):")
pprint(gradients_are_zero)
print(f"\tGradients are NAN:")
pprint(gradients_are_nan)


"""------------------"""
""" Logging Utils """
"""------------------"""

def weights_to_img(w):
    w_pos = np.where(w > 0.0, np.abs(w), 0.0)
    w_neg = np.where(w < 0.0, np.abs(w), 0.0)
    w_max = np.max(np.abs(w))

    blue = w_pos / w_max
    red = w_neg / w_max
    green = np.zeros_like(red)

    return np.stack([red, green, blue], axis=-1)
    
@jit
def compute_activation_imgs(outs, params):
    """Compute metrics from model outputs."""
    img_dict = {}

    N_MAX = 8

    for l_idx, l_out in enumerate(outs):
        # # binary activations
        for k in ["y", "z"]:
            
            # select only a few outputs (e.g., 8) from the batch size
            _v = l_out[k][:N_MAX]

            # add last channel (grayscale)
            _v = _v.reshape((*_v.shape, 1))

            img_dict.update(
                {f"activation_{l_idx}_{k}/{img_i}": img for img_i, img in enumerate(_v)}
            )

        # Add encoder weights as images
        w = params["layers_" + str(l_idx)]["conv"]["kernel"]
        w = w.reshape((-1, w.shape[-1]))
        img_dict[f"weights_{l_idx}"] = weights_to_img(w)

    return img_dict

"""------------------"""
""" Training """
"""------------------"""

# tensorboard writer
log_folder = os.path.join(
    model_folder,
    "logs",
    datetime.now().strftime("%Y%m%d_%H%M%S"),
)
writer = SummaryWriter(log_dir=log_folder)

print("\nTraining")

try:

    # take images from dataloader_original, perform a forward pass, and save to tensorboard as image
    xs_original, labels_original = next(iter(dataloader_val))
    # BATCH_NORM - change here
    # outs, _ = forward_eval_jitted(state["variables"], xs_original)
    outs, o_aux = forward_eval_jitted(state["variables"], xs_original, key)
    # convert to images
    img_dict = compute_activation_imgs(outs, state["variables"]["params"])
    for k, v in img_dict.items():
        v = onp.array(v) # important, tensorboard wants lists or numpy arrays, not jax.numpt arrays
        writer.add_image(k, v, global_step=0, dataformats="HWC")

    for epoch in tqdm(range(model_config["training"]["epochs"])):

        epoch_metrics = {k: 0.0 for k in metrics.keys()}
        epoch_metrics_val = {k: 0.0 for k in metrics_val.keys()}
        best_val_epoch = {
            "epoch": None,
            "metrics": {k: None for k in metrics_val.keys()},
        }

        for batch_idx, (xs, labels) in tqdm(
            enumerate(dataloader_train), leave=False, total=len(dataloader_train)
        ):

            key, subkey = jax.random.split(key, 2)
            batch = {
                "x_1": xs,
                "key_1": key,
                "x_2": xs,
                "key_2": subkey,
                "labels": labels,
            }
            state, metrics, grads = train_step(state, batch)
            LAST_STEP = state["step"]#.item()

            # Log stats for the train batch
            for metric, value in metrics.items():
                writer.add_scalar(
                    "train_batch/" + metric, value.item(), global_step=LAST_STEP
                )
                epoch_metrics[metric] += value

            # test on validation set
            if batch_idx % model_config["validation"]["eval_interval"] == 0:
                xs, labels_val = next(iter(dataloader_val))
                key, subkey = jax.random.split(key, 2)
                batch_val = {
                    "x_1": xs,
                    "key_1": key,
                    "x_2": xs,
                    "key_2": subkey,
                    "labels": labels_val,
                }
                metrics_val = eval_step(state, batch_val)
                for metric, value in metrics_val.items():
                    writer.add_scalar(
                        "val_batch/" + metric, value.item(), global_step=LAST_STEP
                    )
                    epoch_metrics_val[metric] += value

        # Take the average of the epoch stats
        epoch_metrics = jax.tree.map(lambda x: x / (batch_idx + 1), epoch_metrics)
        epoch_metrics_val = jax.tree.map(
            lambda x: x
            / ((batch_idx + 1) // model_config["validation"]["eval_interval"]),
            epoch_metrics_val,
        )
        epoch_step = int(LAST_STEP / (batch_idx + 1))

        # Log epoch stats
        for metric, value in epoch_metrics.items():
            writer.add_scalar(
                metric + "/train", value.item(), global_step=epoch_step
            )
        for metric, value in epoch_metrics_val.items():
            writer.add_scalar(
                metric + "/val", value.item(), global_step=epoch_step
            )

        # Save checkpoint
        if model_config["training"]["checkpoint"]["save"]:
            # save checkpoint
            checkpoint_manager.save(
                step=epoch_step,
                args=orbax.checkpoint.args.StandardSave(state),
            )

        # take images from dataloader_original, perform a forward pass, and save to tensorboard as image
        xs_original, labels_original = next(iter(dataloader_val))
        # BATCH_NORM - change here
        # outs, _ = forward_eval_jitted(state["variables"], xs_original)
        key, _ = jax.random.split(key, 2)
        outs, o_aux = forward_eval_jitted(state["variables"], xs_original, key)
        # convert to images
        img_dict = compute_activation_imgs(outs, state["variables"]["params"])
        for k, v in img_dict.items():
            v = onp.array(v) # important, tensorboard wants lists or numpy arrays, not jax.numpt arrays
            writer.add_image(k, v, global_step=epoch_step, dataformats="HWC")

except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()
