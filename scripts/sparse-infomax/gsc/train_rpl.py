import os
import argparse



default_model = "rpl"
default_number = "1"
default_cuda = "0"

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
print(key)
print(key.shape)
# Init recurrent state
s0 = model.init_state_from_input(key, x_seq[..., 0, :])
pprint(get_shapes(s0))

# Init params and batch_stats
variables = model.init(key, x_seq[..., 0, :], s0)

print(f"\tDict of variables: \n\t{variables.keys()}")

pprint(get_shapes(variables))


"""---------------------"""
""" Forward Pass """
"""---------------------"""

print("\nForward pass jitted")


def forward(variables, xs, key):
    # init state
    s0 = model.init_state_from_input(key, xs[..., 0, :])
    # scan over the sequence
    out = model.apply(
        variables,
        xs,
        s0,
        method=model.scan,
    )
    return out


def forward_eval(variables, xs, key):
    # init state
    s0 = model.init_state_from_input(key, xs[..., 0, :])
    # scan over the sequence
    out = model.apply(
        variables,
        xs,
        s0,
        method=model.scan,
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
outs = forward_jitted(variables, xs, key)
_ = forward_eval_jitted(variables, xs, key)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))

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

"""---------------------"""
""" Loss Function """
"""---------------------"""

sim_fn = partial(
    sbdr.config_similarity_dict[model_config["training"]["loss"]["sim_fn"]["type"]],
    **model_config["training"]["loss"]["sim_fn"]["kwargs"],
)

def flo_loss(
    outs_layer,
):

    eps = model_config["training"]["loss"]["sim_fn"]["kwargs"]["eps"]

    a_fwd = outs_layer["a_fwd"]
    z_fwd = outs_layer["z_fwd"]
    a_rec = outs_layer["a_rec"]
    z_rec = outs_layer["z_rec"]
    a_pred = outs_layer["a_pred"]
    z_pred = outs_layer["z_pred"]

    za_fwd = z_fwd*a_fwd
    za_rec = z_rec*a_rec
    za_pred = z_pred*a_pred

    za_fwd_avg = za_fwd.reshape((-1, za_fwd.shape[-1])).mean(0)
    za_rec_avg = za_rec.reshape((-1, za_rec.shape[-1])).mean(0)
    za_pred_avg = za_pred.reshape((-1, za_pred.shape[-1])).mean(0)

    # # # Forward and prediction infoNCE
    # Positive sample is shifted in time
    p_ii_pred_fwd = (za_pred[..., :-1, :] * za_fwd[..., 1:, :]).sum(-1) + eps
    p_avg_pred_fwd = (za_pred[..., :-1, :] * za_fwd_avg).sum(-1) + eps
    p_avg_fwd_pred = (za_fwd[..., 1:, :] * za_pred_avg).sum(-1) + eps

    flo_pred_fwd_val = -np.log(p_ii_pred_fwd / p_avg_pred_fwd)
    flo_fwd_pred_val = -np.log(p_ii_pred_fwd / p_avg_fwd_pred)
    flo_pred_val = 0.5 * (flo_pred_fwd_val.mean() + flo_fwd_pred_val.mean())

    # # # Recurrent InfoNCE (separate all latent states)
    p_ii_rec = (za_rec * za_rec).sum(-1) + eps
    p_avg_rec = (za_rec * za_rec_avg).sum(-1) + eps

    flo_rec_val = -np.log(p_ii_rec / p_avg_rec)
    flo_rec_val = flo_rec_val.mean()

    # Compute vumulative loss
    flo_val = flo_pred_val + flo_rec_val

    aux = {
        "flo_pred": flo_pred_val,
        "flo_rec": flo_rec_val,
    }

    return flo_val, aux


flo_loss_jitted = jit(flo_loss)

# test loss function
xs, labels = next(iter(dataloader_train))

key = jax.random.key(model_config["model"]["seed"])
key_1, key_2 = jax.random.split(key)

# BATCH_NORM - change here
# outs_1, _ = forward(variables, xs_1, key_1)
# outs_2, _ = forward(variables, xs_2, key_2)
outs_1 = forward(variables, xs[:,1:], key_1)
outs_2 = forward(variables, xs[:,:-1], key_2)

loss = flo_loss_jitted(outs_1[model_config["training"]["layer_idx"]])

print(f"\tFLO Loss: {loss}")

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


"""---------------------"""
""" Utils """
"""---------------------"""

print("\nTraining and Evaluation Steps")

Q_KEYS = ["0.05", "0.25", "0.5", "0.75", "0.95"]
QS = np.array((0.05, 0.25, 0.5, 0.75, 0.95))

@jit
def compute_metrics(outs):
    """Compute metrics from model outputs."""
    metrics = {}

    for l_idx, l_out in enumerate(outs):
        for k in ["z_fwd", "a_fwd", "z_rec", "a_rec","z_pred", "a_pred"]:
            
            # per-unit average
            unit_avg = np.mean(l_out[k].reshape((-1, l_out[k].shape[-1])), axis=0)
            qs_unit_val = np.quantile(unit_avg, QS)

            # per-sample average
            sample_avg = np.mean(l_out[k].reshape((-1, l_out[k].shape[-1])), axis=-1)
            qs_sample_val = np.quantile(sample_avg, QS)


            metrics.update(
                {f"unit_{k}_{l_idx}/{kk}": v for kk, v in zip(Q_KEYS, qs_unit_val)}
            )
            metrics.update(
                {f"sample_{k}_{l_idx}/{kk}": v for kk, v in zip(Q_KEYS, qs_sample_val)}
            )

    return metrics

"""---------------------"""
""" Training and Evaluation Steps """
"""---------------------"""

print("\nTraining and Evaluation Steps")


@jit
def train_step(state, batch):

    # index of the layer that we train
    l_index = model_config["training"]["layer_idx"]
    
    def loss_fn(params):
        # Apply the model
        # BATCH_NORM - change here
        # outs_1, mutable_updates_1 = forward(
        outs = forward(
            {
                "params": params,
                # # BATCH_NORM - change here
                # "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x_1"],
            batch["key_1"],
        )

        # select only outputs from the layer that is being trained
        outs_layer = outs[l_index]
        # outs_1 = jax.tree.map(lambda x : x[:, :-1], outs_layer)
        # outs_2 = jax.tree.map(lambda x : x[:, 1:], outs_layer)

        # Compute FLO loss
        flo_loss_val, aux = flo_loss(outs_layer)
        loss_val = flo_loss_val

        # add small positive pressure on all pre-activations
        exc_fwd = outs_layer["a_fwd"]
        exc_rec = outs_layer["a_rec"]
        exc_pred = outs_layer["a_pred"]
        loss_val = loss_val - model_config["training"]["exc_weight"] * (exc_fwd + exc_rec + exc_pred).mean()

        
        # compute useful metrics for logging
        metrics = compute_metrics(outs)
        # add flo and total loss
        for k, v in aux.items():
            metrics.update({f"aux/{k}": v})
        metrics.update({f"loss/{l_index}": loss_val})

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
    state["variables"]["params"] = optax.apply_updates(
        state["variables"]["params"], updates
    )
    # Update optimizer state
    state["opt_state"] = opt_state

    # # BATCH_NORM - change here
    # # update batch stats
    # state["variables"]["batch_stats"] = others["mutable_updates"]["batch_stats"]

    # update step
    state["step"] += 1

    return state, metrics, grads


@jit
def eval_step(state, batch):

    l_index = model_config["training"]["layer_idx"]

    def loss_fn(params):
        # Apply the model
        # BATCH_NORM - change here
        # outs_1, mutable_updates_1 = forward_eval(
        outs = forward_eval(
            {
                "params": params,
                # # BATCH_NORM - change here
                # "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x_1"],
            batch["key_1"],
        )
        
        # select only outputs from the layer that is being trained
        outs_layer = outs[l_index]
        # outs_1 = jax.tree.map(lambda x : x[:, :-1], outs_layer)
        # outs_2 = jax.tree.map(lambda x : x[:, 1:], outs_layer)

        # Compute FLO loss
        flo_loss_val, aux = flo_loss(outs_layer)
        loss_val = flo_loss_val

        metrics = compute_metrics(outs)
        # add flo and total loss
        for k in aux.keys():
            metrics.update({f"aux/{k}": aux[k]})
        metrics.update({f"loss/{l_index}": loss_val})

        others = {
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
    # "x_2": xs_2,
    # "key_2": key_2,
}

state, metrics, grads = train_step(state, batch)
metrics_val = eval_step(state, batch)

print(f"\tmetrics: {metrics}")

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


def activation_to_img(z, normalize=True):
    
    # # Compute Width and Height to get an approximate square shape, padding if necessary
    # n_height = np.sqrt(z.shape[-1]).astype(int)
    # n_width = z.shape[-1] // n_height + int(not (z.shape[-1] % n_height) == 0)
    # # normalize min-max in [0, 1]
    # if normalize:
    #     z = (z - z.min()) / (z.max() - z.min())
    # # pad z with zero if necessary, so we can then reshape the feature dimension (last)
    # # to the given width and height
    # pad_width = [(0, 0)] * (len(z.shape) - 1)
    # pad_width.append((0, n_height * n_width - z.shape[-1]))
    # z = np.pad(z, pad_width, mode="constant", constant_values=0.0)
    # # add a dummy final channel dimension, like it's grayscale
    # z = z.reshape((*z.shape[:-1], n_height, n_width, 1))

    # just keep as is, as an alternative, adding the final channel dimension
    z = z.reshape((*z.shape, 1))


    return onp.array(z)


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
    outs = forward_eval_jitted(state["variables"], xs_original)
    for l_idx, l_out in enumerate(outs):
        activation_img = activation_to_img(l_out["z"])
        for i in range(min(16, activation_img.shape[0])):
            writer.add_image(
                f"activation_img_{l_idx}/{i+1}",
                activation_img[i],
                global_step=0,
                dataformats="HWC",
            )

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

            key, _ = jax.random.split(key, 2)
            batch = {
                "x_1": xs, # jax.device_put(xs_1),
                "key_1": key,
            }
            state, metrics, grads = train_step(state, batch)
            LAST_STEP = state["step"].item()

            # Log stats for the train batch
            for metric, value in metrics.items():
                writer.add_scalar(
                    "train_batch/" + metric, value.item(), global_step=LAST_STEP
                )
                epoch_metrics[metric] += value

            # test on validation set
            if batch_idx % model_config["validation"]["eval_interval"] == 0:
                xs, labels_val = next(iter(dataloader_val))
                batch_val = {
                    "x_1": xs,
                    # "x_2": xs_2_val,
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
        outs = forward_eval_jitted(state["variables"], xs_original)
        for l_idx, l_out in enumerate(outs):
            activation_img = activation_to_img(l_out["z"])
            for i in range(min(16, activation_img.shape[0])):
                writer.add_image(
                    f"activation_img_{l_idx}/{i+1}",
                    activation_img[i],
                    global_step=epoch_step,
                    dataformats="HWC",
                )

except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()
