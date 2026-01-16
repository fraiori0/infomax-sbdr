import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "3"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"

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
BINARIZE_THRESHOLD = 0.5  # threshold for binarization, only used if BINARIZE is True

default_model = "vgg_sigmoid_and"
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
""" Import Decoder Config"""
"""---------------------"""

decoder_model_folder = os.path.join(
    model_folder,
    "decoder",
)

print("\nImport decoder config")

# import the elman_config.toml
with open(os.path.join(decoder_model_folder, "config.toml"), "r") as f:
    decoder_model_config = toml.load(f)

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
        1 - decoder_model_config["validation"]["split"],
        decoder_model_config["validation"]["split"],
    ],
    generator=torch.Generator().manual_seed(decoder_model_config["model"]["seed"]),
)


dataloader_train = sbdr.NumpyLoader(
    dataset_train,
    batch_size=decoder_model_config["training"]["batch_size"],
    shuffle=False,
    drop_last=False,
)

dataloader_val = sbdr.NumpyLoader(
    dataset_val,
    batch_size=decoder_model_config["validation"]["dataloader"]["batch_size"],
    shuffle=True,
    drop_last=False,
)

xs, labels = next(iter(dataloader_train))

print(f"\tInput xs: {xs.shape} (dtype: {xs.dtype})")
print(f"\tLabels: {labels.shape} (dtype: {labels.dtype})")

def unnormalize(xs):
    return (
        xs * decoder_model_config["dataset"]["transform"]["normalization"]["std"]
        + decoder_model_config["dataset"]["transform"]["normalization"]["mean"]
    )

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
print(f"\tConv shape: {outs['conv_shape']}")


"""---------------------"""
""" Init Decoder """
"""---------------------"""

conv_shape = tuple([s.item() for s in outs["conv_shape"]])


print("\nInit Decoder Model")

decoder_model_class = sbdr.config_module_dict[decoder_model_config["model"]["type"]]

decoder_model = decoder_model_class(
    **decoder_model_config["model"]["kwargs"],
    activation_fn=sbdr.config_activation_dict[decoder_model_config["model"]["activation"]],
    preconv_chw = conv_shape,
    training=True,
)

decoder_model_eval = decoder_model_class(
    **decoder_model_config["model"]["kwargs"],
    activation_fn=sbdr.config_activation_dict[decoder_model_config["model"]["activation"]],
    preconv_chw = conv_shape,
    training=False,
)

# # # Initialize parameters
# Take some data
xs, labels = next(iter(dataloader_train))
# Encode using pre-trained model
outs, _ = forward_eval_jitted(variables, xs)
# Generate key
key = jax.random.key(decoder_model_config["model"]["seed"])
# Init params and batch_stats
decoder_variables = decoder_model.init(key, outs["z"])


print(f"\tDict of variables: \n\t{decoder_variables.keys()}")


"""---------------------"""
""" Loss Function """
"""---------------------"""

print("\nForward pass and loss function")

def decoder_forward(variables, xs, key):
    return decoder_model.apply(
        variables,
        xs,
        # DROPOUT
        # key for dropout
        rngs={"dropout": key},
        # BATCH_NORM
        # batch stats should be updated
        mutable=["batch_stats"],
    )

def decoder_forward_eval(variables, xs):
    return decoder_model_eval.apply(
        variables,
        xs,
        # BATCH_NORM
        # batch stats should be updated
        mutable=["batch_stats"],
    )

decoder_forward_jitted = jit(decoder_forward)
decoder_forward_eval_jitted = jit(decoder_forward_eval)

def reconstruction_loss(decoder_outs, xs):
    # Compute Mean Square Error compared to the original
    return ((decoder_outs["z"] - xs) ** 2).mean()

# test the forward pass and loss function

xs, labels = next(iter(dataloader_train))
# Forward pass through pre-trained encoder model
outs, _ = forward_eval_jitted(variables, xs)
# Forward pass through decoder
key = jax.random.key(decoder_model_config["model"]["seed"])
decoder_outs, decoder_mutable_updates = decoder_forward_jitted(decoder_variables, outs["z"], key)


print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(decoder_outs))

print(f"\tMutable updates:")
pprint(get_shapes(decoder_mutable_updates))

# Compute loss
loss = reconstruction_loss(decoder_outs, xs)
print(f"\tLoss: {loss}")


"""---------------------"""
""" Checkpointing """
"""---------------------"""

print("\nDecoder Checkpointing Stuff")

decoder_checkpoint_manager = orbax.checkpoint.CheckpointManager(
    directory=os.path.join(decoder_model_folder, "checkpoints"),
    # checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
    options=orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=decoder_model_config["training"]["checkpoint"]["save_interval"],
        max_to_keep=decoder_model_config["training"]["checkpoint"]["max_to_keep"],
        step_format_fixed_length=5,
    ),
)

# # # Initialize the optimizer as well, to properly restore the full checkpoint
decoder_optimizer = sbdr.config_optimizer_dict[decoder_model_config["training"]["optimizer"]["type"]]
decoder_optimizer = decoder_optimizer(**decoder_model_config["training"]["optimizer"]["kwargs"])

decoder_state = {
    "variables": decoder_variables,
    "opt_state": decoder_optimizer.init(decoder_variables["params"]),
    "step": 0,
}

LAST_STEP = decoder_checkpoint_manager.latest_step()
if LAST_STEP is None:
    LAST_STEP = 0
else:
    decoder_state = decoder_checkpoint_manager.restore(
        step=LAST_STEP, args=orbax.checkpoint.args.StandardRestore(decoder_state)
    )


print(f"\tLast step: {LAST_STEP}")


"""---------------------"""
""" Training and Evaluation Steps """
"""---------------------"""

print("\nTraining and Evaluation Steps")

@jit
def decoder_train_step(decoder_state, batch):

    def loss_fn(params):
        # Apply the models
        outs, _ = forward_eval(variables, batch["x"])
        decoder_outs, mutable_updates = decoder_forward(
            {"params": params},
            xs = outs["z"],
            key = batch["key"],
        )
        # Compute loss
        loss_val = reconstruction_loss(decoder_outs, batch["x"])
        metrics = {
            "loss/mse": loss_val,
        }
        return loss_val, (metrics, {"mutable_updates": mutable_updates})

    # compute gradient, loss, and aux
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss_val, (metrics, others)), grads = grad_fn(decoder_state["variables"]["params"])
    # update weights
    updates, opt_state = decoder_optimizer.update(grads, decoder_state["opt_state"], decoder_state["variables"]["params"])
    decoder_state["variables"]["params"] = optax.apply_updates(
        decoder_state["variables"]["params"], updates
    )
    # Update optimizer state
    decoder_state["opt_state"] = opt_state

    # BATCH_NORM - change here
    # update batch stats
    decoder_state["variables"]["batch_stats"] = others["mutable_updates"]["batch_stats"]

    # update step
    decoder_state["step"] += 1
    return decoder_state, metrics, grads

@jit
def decoder_eval_step(decoder_state, batch):

    def loss_fn(params):
        # Apply the models
        outs, _ = forward_eval(variables, batch["x"])
        decoder_outs, _ = decoder_forward_eval(
            {"params": params},
            xs = outs["z"],
        )
        # Compute loss
        loss_val = reconstruction_loss(decoder_outs, batch["x"])
        metrics = {
            "loss/mse": loss_val,
        }
        return loss_val, metrics

    loss_val, metrics = loss_fn(decoder_state["variables"]["params"])

    return metrics

print("\tTest one train step and one eval step")

xs, labels = next(iter(dataloader_train))
key, _ = jax.random.split(key)
batch = {
    "x": xs,
    "key": key,
}
decoder_state, metrics, grads = decoder_train_step(decoder_state, batch)
print(metrics)
metrics_val = decoder_eval_step(decoder_state, batch)
print(metrics_val)


gradients_are_finite = jax.tree_util.tree_map(
    lambda x: np.all(np.isfinite(x)).item(), grads
)
gradients_are_zero = jax.tree_util.tree_map(
    lambda x: np.all(np.isclose(x, 0.0, rtol=1e-20, atol=1e-20)).item(), grads
)
gradients_are_nan = jax.tree_util.tree_map(lambda x: np.any(np.isnan(x)).item(), grads)
print(f"\tGradients are FINITE:")
pprint(gradients_are_finite)
print(f"\tGradients are ZERO:")
pprint(gradients_are_zero)
print(f"\tGradients are NAN:")
pprint(gradients_are_nan)


"""------------------"""
""" Training """
"""------------------"""

print("\nTraining")

key = jax.random.key(decoder_model_config["model"]["seed"])

# tensorboard writer
decoder_log_folder = os.path.join(
    decoder_model_folder,
    "logs",
    datetime.now().strftime("%Y%m%d_%H%M%S"),
)
writer = SummaryWriter(log_dir=decoder_log_folder)

try:
    for epoch in tqdm(range(decoder_model_config["training"]["epochs"])):
        
        epoch_metrics = {k: 0.0 for k in metrics.keys()}
        epoch_metrics_val = {k: 0.0 for k in metrics_val.keys()}

        for batch_idx, (xs, labels) in tqdm(enumerate(dataloader_train), leave=False, total=len(dataloader_train)):
            key, _ = jax.random.split(key)
            batch = {
                "x": xs,
                "key": key,
            }
            decoder_state, metrics, grads = decoder_train_step(decoder_state, batch)

            LAST_STEP = decoder_state["step"].item()

            # Log stats for the train batch
            for metric, value in metrics.items():
                writer.add_scalar(
                    f"train/batch/{metric}",
                    value.item(),
                    LAST_STEP,
                )
                epoch_metrics[metric] += value

            # test on validation set
            if batch_idx % decoder_model_config["validation"]["eval_interval"] == 0:
                xs_val, labels_val = next(iter(dataloader_val))
                key, _ = jax.random.split(key)
                batch_val = {
                    "x": xs_val,
                    "key": key,
                }
                metrics_val = decoder_eval_step(decoder_state, batch_val)
                for metric, value in metrics_val.items():
                    writer.add_scalar(
                        f"val/batch/{metric}", value.item(), global_step=LAST_STEP
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
                f"train/epoch/{metric}", value.item(), global_step=epoch_step
            )
        for metric, value in epoch_metrics_val.items():
            writer.add_scalar(
                f"val/epoch/{metric}", value.item(), global_step=epoch_step
            )

        # Save checkpoint
        if decoder_model_config["training"]["checkpoint"]["save"]:
            # save checkpoint
            decoder_checkpoint_manager.save(
                step=epoch_step,
                args=orbax.checkpoint.args.StandardSave(decoder_state),
            )
        
        # After each epoch, take a few images, original and reconstructed, to show
        # take images from dataloader_original, perform a forward pass, and save to tensorboard as image
        outs, _ = forward_eval(variables, xs_val)
        decoder_outs, _ = decoder_forward_eval_jitted(
            decoder_state["variables"],
            xs = outs["z"],
        )

        for i in range(5):
            writer.add_image(
                f"img/original/{i+1}",
                onp.array(xs_val[i]),
                global_step=epoch_step,
                dataformats="HWC",
            )
            writer.add_image(
                f"img/reconstructed/{i+1}",
                onp.array(decoder_outs["z"][i]),
                global_step=epoch_step,
                dataformats="HWC",
            )


except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()