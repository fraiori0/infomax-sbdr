import os
import argparse


default_model = "fashionmnist"
default_number = "1"

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
    "ahtd",
    args.model,
    args.number,
)


# model_config = sbdr.ahtd.load_model_config(os.path.join(model_folder, "config.toml"))

# import the elman_config.toml
with open(os.path.join(model_folder, "config.toml"), "r") as f:
    config_dict = toml.load(f)

print(f"\nLoaded model config from:\n\t{model_folder}")
pprint(config_dict)


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
    **config_dict["dataset"]["kwargs"],
)

dataset_train, dataset_val = torch.utils.data.random_split(
    dataset,
    [
        1 - config_dict["validation"]["split"],
        config_dict["validation"]["split"],
    ],
    generator=torch.Generator().manual_seed(config_dict["model"]["seed"]),
)


dataloader_train = sbdr.NumpyLoader(
    dataset_train,
    batch_size=config_dict["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
    # num_workers=1,
)

dataloader_val = sbdr.NumpyLoader(
    dataset_val,
    batch_size=config_dict["training"]["batch_size"],
    shuffle=False,
    drop_last=True,
    # num_workers=2,
    generator=torch.Generator().manual_seed(config_dict["model"]["seed"]),
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
""" Visualize the images"""
"""---------------------"""


# def untransform(x):
#     return (x * np.array([0.3530])) + np.array([0.2860])


# # visualize the images
# # converted to BGR
# for i, (img_1, img_2) in enumerate(zip(xs_1, xs_2)):
#     if i>=5:
#         break
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
#     # Save Image
#     cv2.imwrite(f"image_1_{i}.png", img_1)
#     cv2.imwrite(f"image_2_{i}.png", img_2)

#     # # show image
#     # cv2.imshow("Image 1", img_1)
#     # cv2.imshow("Image 2", img_2)
#     # k = cv2.waitKey(0)
#     # if k == ord("q"):
#     #     break

# # cv2.destroyAllWindows()


"""---------------------"""
""" Init Network """
"""---------------------"""

print("\nInitializing model")

print("Building model...")
key = jax.random.key(config_dict['model']['seed'])
model = sbdr.ahtd.build_model(config_dict, key)
pprint(model)

"""---------------------"""
""" Checkpointing """
"""---------------------"""
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    directory=os.path.join(model_folder, "checkpoints"),
    # checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
    options=orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=config_dict["training"]["checkpoint_interval"],
        max_to_keep=config_dict["training"]["checkpoint_max_to_keep"],
        step_format_fixed_length=4,
    ),
)

print("\nCheckpointing stuff")

state = {
    "model": model,
    "step": 0,
}

# # save state, to debug saving and loading
# checkpoint_manager.save(
#     0,
#     args=orbax.checkpoint.args.PyTreeSave(state),
# )

LAST_STEP = checkpoint_manager.latest_step()
if LAST_STEP is None:
    LAST_STEP = 0
else:
    state = checkpoint_manager.restore(
        step=LAST_STEP, args=orbax.checkpoint.args.PyTreeRestore(state)
    )

model = state["model"]

print(f"\tLast step: {LAST_STEP}")


"""---------------------"""
""" Forward Pass """
"""---------------------"""

print("\nForward scan jitted")

@partial(jit, static_argnums=(2, 3))
def forward(xs, params, hyperparams, config):
    model_state = sbdr.ahtd.init_state_from_input(
        xs,
        params,
        hyperparams,
        config,
    )
    return sbdr.ahtd.forward_stack(
        model_state,
        xs,
        params,
        hyperparams,
        config,
    )

for k in model.keys():
    print(k, type(model[k][0]))

# test the forward scan
xs, labels = next(iter(dataloader_train))
outs = forward(xs, **model)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(sbdr.get_shapes(outs))

"""---------------------"""
""" Parameter Update """
"""---------------------"""

print("\nParameter update jitted")

@partial(jit, static_argnums=(2, 3))
def update_params(outs, params, hyperparams, config):
    new_params =  sbdr.ahtd.update_stack(
        outs,
        params,
        hyperparams,
        config,
    )
    return new_params


new_params = update_params(outs, **model)

print(f"\tOutput shapes:")
pprint(sbdr.get_shapes(outs))
print("\tShapes of new params:")
pprint(sbdr.get_shapes(new_params))

"""---------------------"""
""" Test an epoch """
"""---------------------"""

for xs, labels in dataloader_train:
    outs = forward(xs, **model)
    new_params = update_params(outs, **model)
    model = sbdr.ahtd.AHTDModule(new_params, model["hyperparams"], model["config"])
    z = outs[0]["z"]
    td_err = outs[0]["td_error"]
    print(z.mean(), z.std(), td_err.mean())

exit()
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

    for i, out_layer in enumerate(outs):
        # compute quantiles of unit activation in the batch
        z_avg = np.mean(out_layer["z"].reshape((-1, out_layer["z"].shape[-1])), axis=0)
        qs_z_val = np.quantile(z_avg, QS)

        # compute quantiles of sample activation in the batch
        s_avg = np.mean(out_layer["z"].reshape((-1, out_layer["z"].shape[-1])), axis=-1)
        qs_s_val = np.quantile(s_avg, QS)

        # compute average td error
        td_err = np.abs(out_layer["td_error"]).mean()

        metrics.update(
            {f"{i}_unit/{k}": v for k, v in zip(Q_KEYS, qs_z_val)}
        )
        metrics.update(
            {f"{i}_sample/{k}": v for k, v in zip(Q_KEYS, qs_s_val)}
        )

        metrics[f"td_error/{i}"] = td_err

    return metrics

"""---------------------"""
""" Training and Evaluation Steps """
"""---------------------"""

@partial(jit, static_argnums=(2, 3))
def train_step_jit_part(batch, params, hyperparams, config):
    # Forward pass
    outs = forward(batch["x"], params, hyperparams, config)
    # remove the first steps on the time dimension, to have some warm-up
    outs = jax.tree.map(lambda x: x[..., config_dict["model"]["seq"]["skip_first"]:, :], outs)
    # Update params
    new_params = update_params(outs, params, hyperparams, config)

    # Metrics
    metrics = compute_metrics(outs)

    return new_params, metrics

# this function avoid issues with JIT in not using strings as output
def train_step(batch, state):
    params, metrics = train_step_jit_part(batch, **state["model"])
    # create new state
    state = {
        "model": sbdr.ahtd.AHTDModule(
            params,
            state["model"]["hyperparams"],
            state["model"]["config"],
        ),
        "step": state["step"] + 1
    }
    return state, metrics


@partial(jit, static_argnums=(2, 3))
def eval_step_jit_part(batch, params, hyperparams, config):
    # Forward pass
    outs = forward(batch["x"], params, hyperparams, config)
    # remove the first steps on the time dimension, to have some warm-up
    outs = jax.tree.map(lambda x: x[..., config_dict["model"]["seq"]["skip_first"]:, :], outs)
    # Metrics
    metrics = compute_metrics(outs)

    return metrics

def eval_step(batch, state):
    metrics = eval_step_jit_part(batch, **state["model"])
    return metrics

print("\t Test one train step and one eval step")

xs, labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
key = jax.random.key(config_dict["model"]["seed"])
batch = {
    "x": xs,
    "key": key,
}

state, metrics = train_step(batch, state)
metrics_val = eval_step(batch, state)

print(f"\tMetrics:")
pprint(metrics)

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

def outs_to_image_dict(outs):
    n_img = 16
    img_dict = {}
    for i, out_layer in enumerate(outs):
        # take activations
        z = out_layer["z"]
        # convert to img the first n_img activations
        z_img = activation_seq_to_img(z[:n_img])
        # save to dict
        for j in range(n_img):
            img_dict[f"{i}_activity/{j}"] = z_img[j]

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

    for epoch in tqdm(range(config_dict["training"]["epochs"])):

        epoch_metrics = {k: 0.0 for k in metrics.keys()}
        epoch_metrics_val = {k: 0.0 for k in metrics_val.keys()}

        for batch_idx, (xs, labels) in tqdm(
            enumerate(dataloader_train), leave=False, total=len(dataloader_train)
        ):

            key, _ = jax.random.split(key)
            batch = {
                "x": xs,
                "key": key,
            }
            state, metrics = train_step(batch, state)
            LAST_STEP = state["step"]#.item()

            # Log stats for the train batch
            for metric, value in metrics.items():
                writer.add_scalar(
                    metric + "/train/batch", value.item(), global_step=LAST_STEP
                )
                epoch_metrics[metric] += value

            # test on validation set
            if batch_idx % config_dict["validation"]["interval"] == 0:
                xs_val, labels_val = next(iter(dataloader_val))
                key_val, _ = jax.random.split(jax.random.key(batch_idx))
                batch_val = {
                    "x": xs_val,
                    "key": key_val,
                }
                metrics_val = eval_step(batch_val, state)
                for metric, value in metrics_val.items():
                    writer.add_scalar(
                        metric + "/val/batch", value.item(), global_step=LAST_STEP
                    )
                    epoch_metrics_val[metric] += value

        # Take the average of the epoch stats
        epoch_metrics = jax.tree.map(lambda x: x / (batch_idx + 1), epoch_metrics)
        epoch_metrics_val = jax.tree.map(
            lambda x: x
            / ((batch_idx + 1) // config_dict["validation"]["interval"]),
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
        if config_dict["training"]["save"]:
            # save checkpoint
            checkpoint_manager.save(
                step=epoch_step,
                args=orbax.checkpoint.args.PyTreeSave(state),
            )

        # take images from dataloader_original, perform a forward pass, and save to tensorboard as image
        xs_img, labels_img = next(iter(dataloader_val))
        outs_img = forward(xs_img, **state["model"])
        # convert to images
        img_dict = outs_to_image_dict(outs_img)
        for k, v in img_dict.items():
            writer.add_image(
                k,
                v,
                global_step=epoch_step,
                dataformats="HWC",
            )


except KeyboardInterrupt:
    print("\nTraining interrupted")

checkpoint_manager.save(
    step=epoch_step,
    args=orbax.checkpoint.args.PyTreeSave(state),
)

writer.close()

print("\nDone.")

