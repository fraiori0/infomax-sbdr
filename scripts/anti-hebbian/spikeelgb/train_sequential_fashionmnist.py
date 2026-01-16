import os
import argparse

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"

default_model = "spikeelgb"
default_number = "1"

default_cuda = "2"

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

# create the transform
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
    shuffle=model_config["training"]["dataloader"]["shuffle"],
    drop_last=model_config["training"]["dataloader"]["drop_last"],
    # num_workers=1,
)

dataloader_val = sbdr.NumpyLoader(
    dataset_val,
    batch_size=model_config["validation"]["dataloader"]["batch_size"],
    shuffle=False,
    drop_last=True,
    # num_workers=2,
)


xs, labels = next(iter(dataloader_train))
# print original device
print(f"\tOriginal device: {xs.device}")
print()
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)

print(f"\tInput xs: {xs.shape} (dtype: {xs.dtype})")
print(f"\tLabels: {labels.shape} (dtype: {labels.dtype})")


"""---------------------"""
""" Init Network """
"""---------------------"""

print("\nInitializing model")

model_class = sbdr.config_ah_module_dict[model_config["model"]["type"]]

th = -0.5
model = model_class(
    **model_config["model"]["kwargs"],
    threshold = [th for _ in model_config["model"]["kwargs"]["n_units"]],
    train = True,
)

model_eval = model_class(
    **model_config["model"]["kwargs"],
    # threshold = [th for _ in model_config["model"]["kwargs"]["n_units"]],
    train = False,
)

# # # Initialize parameters
# Take some data
xs, labels = next(iter(dataloader_train))
# Generate a random initial state
key = jax.random.key(model_config["model"]["seed"])
e0_l = model.gen_initial_state(key, xs[..., 0, :])

print(xs.shape)
print(len(e0_l))
print([e0.shape for e0 in e0_l])

# Init params and batch_stats
variables = model.init(
    key,
    xs[..., 0, :],
    *e0_l,
)

print(f"\tDict of variables: \n\t{variables.keys()}")


# print the shapes nicely
def get_shapes(nested_dict):
    return jax.tree_util.tree_map(lambda x: x.shape, nested_dict)


pprint(get_shapes(variables))

"""---------------------"""
""" Forward Pass """
"""---------------------"""

# perform a single forward pass as a test
outs = model.apply(
    variables,
    xs[..., 0, :],
    *e0_l,
)

print(f"\tOutput shapes for single step forward pass:")
pprint(get_shapes(outs))

print("\nForward scan jitted")


def forward_scan(variables, xs_seq, key):
    e0_l = model.gen_initial_state(key, xs_seq[..., 0, :])
    return model.apply(
        variables,
        xs_seq,
        *e0_l,
        method="scan",
    )


def forward_scan_eval(variables, xs_seq, key):
    e0_l = model_eval.gen_initial_state(key, xs_seq[..., 0, :])
    return model_eval.apply(
        variables,
        xs_seq,
        *e0_l,
        method="scan",
    )
forward_jitted = jit(forward_scan)
forward_eval_jitted = jit(forward_scan_eval)

# test the forward scan
xs, labels = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])

outs = forward_jitted(variables, xs, key)
outs_eval = forward_eval_jitted(variables, xs, key)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))


"""---------------------"""
""" Loss Function """
"""---------------------"""

sim_fn = partial(
    sbdr.config_similarity_dict[model_config["training"]["loss"]["sim_fn"]["type"]],
    **model_config["training"]["loss"]["sim_fn"]["kwargs"],
)

def mi_loss(
        z,
        key,
        u_ii=None,
    ):
    # z of shape [batch, time, features]
    eps = 1.0e-6

    # flatten all dims except features, shuffle, and unflatten
    batch_shape = z.shape[:-1]
    z = z.reshape((-1, z.shape[-1]))
    z = jax.random.permutation(key, z, axis=0)
    # reshape to [time, batch, features], i.e., inverting batch_shape
    z = z.reshape((*batch_shape, -1))
    z = np.moveaxis(z, 1, 0)

    # # # Contrastive Mutual Information FLO estimator
    # # Positive samples
    p_ii = sim_fn(z, z)
    # # Negative samples (on ex reshaped and shuffled batch dim)
    p_ij = sim_fn(z[..., :, None, :], z[..., None, :, :])
    # InfoNCE
    mi = sbdr.infonce(p_ii, p_ij, eps=eps)
    mi = mi.mean()

    # # Average activation
    # z_avg = z.mean(axis=0)
    # z_avg = jax.lax.stop_gradient(z_avg)
    # # alpha = ALPHA
    # alpha = np.where(z_avg < 0.05, 0.0, 1.0 - 0.0)
    # h = (1-alpha) * z*(z+0.5) + alpha * (z-1)*(z-1.5)
    # h = h.mean()
    
    loss_val = - mi
    return loss_val

def module_losses(outs, key):
    # compute the loss for each layer in the module
    loss_vals = []
    for idx_layer, z in enumerate(outs["e_l"]):
        # use this to skip a layer when debugging
        # if idx_layer==1:
        #     continue
        l_val = mi_loss(
            z,
            key,
        )
        loss_vals.append(l_val)
        key, _ = jax.random.split(key)

    return np.array(loss_vals).sum()


mi_loss_jitted = jit(mi_loss)

# test loss function
xs, labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)

key = jax.random.key(model_config["model"]["seed"])

outs = forward_jitted(variables, xs, key)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))

loss = module_losses(outs, key)

print(f"\tLoss: {loss}")


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

LAST_STEP = checkpoint_manager.latest_step()
if LAST_STEP is None:
    LAST_STEP = 0
else:
    state = checkpoint_manager.restore(
        step=LAST_STEP, args=orbax.checkpoint.args.StandardRestore(state)
    )

variables = state["variables"].copy()

print(f"\tLast step: {LAST_STEP}")


"""---------------------"""
""" Training Utils """
"""---------------------"""

print("\nTraining and Evaluation Steps")

Q_KEYS = ["0.05", "0.25", "0.5", "0.75", "0.95"]
QS = np.array((0.05, 0.25, 0.5, 0.75, 0.95))

@jit
def compute_metrics(outs):
    """Compute metrics from model outputs."""

    for idx_layer, z in enumerate(outs["out"]):

        # compute quantiles of unit activation in the batch
        z_avg = np.mean(z.reshape((-1, z.shape[-1])), axis=0)
        qs_z_val = np.quantile(z_avg, QS)

        # compute quantiles of sample activation in the batch
        s_avg = np.mean(z.reshape((-1, z.shape[-1])), axis=-1)
        qs_s_val = np.quantile(s_avg, QS)

        metrics = {
            f"unit/{idx_layer}/{k}": v for k, v in zip(Q_KEYS, qs_z_val)
        }
        metrics.update(
            {f"sample/{idx_layer}/{k}": v for k, v in zip(Q_KEYS, qs_s_val)}
        )

    return metrics

"""---------------------"""
""" Training And Eval Steps """
"""---------------------"""

print("\nTraining and Evaluation Steps")


@jit
def train_step(state, batch):
    def loss_fn(params):
        # Apply the model
        outs = forward_scan(
            {
                "params": params,
                # # BATCH_NORM - change here
                # "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x"],
            batch["key"],
        )

        # Compute FLO loss
        mi_loss_val = module_losses(outs, batch["key"])
        loss_val = mi_loss_val

        metrics = compute_metrics(outs)

        metrics["loss/total"] = loss_val
        metrics["loss/mi"] = mi_loss_val


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
    def loss_fn(params):
        # Apply the model
        # BATCH_NORM - change here
        # outs_1, mutable_updates_1 = forward_eval(
        outs = forward_scan_eval(
            {
                "params": params,
                # # BATCH_NORM - change here
                # "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x"],
            batch["key"],
        )

        # Compute FLO loss
        mi_loss_val = module_losses(outs, batch["key"])
        loss_val = mi_loss_val

        # # Compute metrics
        metrics = compute_metrics(outs)
        metrics["loss/total"] = loss_val
        metrics["loss/mi"] = mi_loss_val

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
    "x": xs,
    "key": key,
}

state, metrics, grads = train_step(state, batch)
metrics_val = eval_step(state, batch)

print(f"\tLoss: {metrics['loss/total']}")

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

def activation_seq_to_img(z_seq):
    # # width and height are the time step and number of features, respectively, kinda like the plot of a neural recording
    # height = z.shape[-1]
    # width = z.shape[-2]
    # add a dummy final channel dimension, like it's grayscale
    z = z_seq[..., None]
    # Convert to NumPy
    z = onp.array(z)
    return z

def img_from_output(outs):
    # for each layer in outs["z"], convert to image
    img_dict = {}
    for idx_layer, z in enumerate(outs["out"]):
        z_img = activation_seq_to_img(z)
        img_dict[f"img_layer_{idx_layer}"] = z_img
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

            key, _ = jax.random.split(key)
            batch = {
                "x": xs,
                "key": key,
            }
            state, metrics, grads = train_step(state, batch)
            LAST_STEP = state["step"]#.item()

            # Log stats for the train batch
            for metric, value in metrics.items():
                writer.add_scalar(
                    "train/" + metric, value.item(), global_step=LAST_STEP
                )
                epoch_metrics[metric] += value

            # test on validation set
            if batch_idx % model_config["validation"]["eval_interval"] == 0:
                xs_val, labels_val = next(iter(dataloader_val))
                key_val, _ = jax.random.split(jax.random.key(batch_idx))
                batch_val = {
                    "x": xs_val,
                    "key": key_val,
                }
                metrics_val = eval_step(state, batch_val)
                for metric, value in metrics_val.items():
                    writer.add_scalar(
                        "val/" + metric, value.item(), global_step=LAST_STEP
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
        xs_img, labels_img = next(iter(dataloader_val))
        key_img = jax.random.key(epoch_step)
        outs_img = forward_eval_jitted(state["variables"], xs_img, key_img)
        # convert to images
        img_dict = img_from_output(outs_img)
        for k, v in img_dict.items():
            activation_imgs = v
            for i in range(8):
                writer.add_image(
                    f"{k}/{i+1}",
                    activation_imgs[i],
                    global_step=epoch_step,
                    dataformats="HWC",
                )


except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()
