import os
import argparse


default_model = "fashionmnist"
default_number = ""
poisson_data = True
default_checkpoint_subfolder = "manual_select" # 
default_step = 50
layer_features = (-2, -1,)

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
    "ahtd",
    args.model,
    args.number,
)


# config_dict = sbdr.ahtd.load_config_dict(os.path.join(model_folder, "config.toml"))

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


if poisson_data:

    dataset = sbdr.FashionMNISTPoissonDataset(
        folder_path=data_folder,
        kind="train",
        transform=None,
        flatten=True,
        **config_dict["dataset"]["kwargs"],
    )
    dataset_test = sbdr.FashionMNISTPoissonDataset(
        folder_path=data_folder,
        kind="test",
        transform=None,
        flatten=True,
        **config_dict["dataset"]["kwargs"],
    )
else:
    # Alternative
    dataset = sbdr.FashionMNISTDataset(
        folder_path=data_folder,
        kind="train",
        transform=None,
        flatten=False,
        sequential=True,
    )

    dataset_test = sbdr.FashionMNISTDataset(
        folder_path=data_folder,
        kind="test",
        transform=None,
        flatten=False,
        sequential=True,
    )

dataloader_train = sbdr.NumpyLoader(
    dataset,
    batch_size=config_dict["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
    # num_workers=1,
)

dataloader_test = sbdr.NumpyLoader(
    dataset_test,
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
    directory=os.path.join(model_folder, default_checkpoint_subfolder),
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

state = checkpoint_manager.restore(
    step=default_step, args=orbax.checkpoint.args.PyTreeRestore(state)
)

model = state["model"].copy()

print(f"\tChkp step: {default_step}")


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
    outs = sbdr.ahtd.forward_stack(
        model_state,
        xs,
        params,
        hyperparams,
        config,
    )
    outs = jax.tree.map(lambda x: x[..., config_dict["model"]["seq"]["skip_first"]:, :], outs)
    return outs

for k in model.keys():
    print(k, type(model[k][0]))

# test the forward scan
xs, labels = next(iter(dataloader_train))
outs = forward(xs, **model)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(sbdr.get_shapes(outs))

# """---------------------"""
# """ Test an epoch """
# """---------------------"""

# for xs, labels in dataloader_train:
#     outs = forward(xs, **model)
#     new_params = update_params(outs, **model)
#     model = sbdr.ahtd.AHTDModule(new_params, model["hyperparams"], model["config"])
#     z = outs[0]["z"]
#     td_err = outs[0]["td_error"]
#     print(z.mean(), z.std(), td_err.mean())


"""---------------------"""
""" Forward pass on training set """
"""---------------------"""

print("\nForward pass on the whole training set")

key = jax.random.key(config_dict["model"]["seed"])

zs = []
labels_categorical = []

for i, (xs, labels) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):

    # # For quicker debugging
    # if i > 20:
    #     break

    model_state = sbdr.ahtd.init_state_from_input(
        xs,
        **model,
    )

    z = sbdr.ahtd.extract_features(
        model_state,
        x_seq = xs,
        layer_idxs = layer_features,
        **model,

    )

    # Record temporal average of output
    zs.append(onp.array(z))

    # And categorical labels
    lab_cat = labels.argmax(axis=-1)
    labels_categorical.append(lab_cat)
    
# Convert to have numpy arrays

zs = onp.concatenate(zs, axis=0)
labels_categorical = onp.concatenate(labels_categorical, axis=0)

history = {
    "zs": zs,
    "labels_categorical": labels_categorical,
}

print(f"\tOutput shapes:")

pprint(sbdr.get_shapes(history))

print(f"\tAverage Activity: {history['zs'].mean()}")


"""---------------------"""
""" Forward pass on test set """
"""---------------------"""

print("\nForward pass on the whole test set")

key = jax.random.key(config_dict["model"]["seed"])

zs_test = []
labels_categorical_test = []

for i, (xs, labels) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):

    # # For quicker debugging
    # if i > 20:
    #     break

    model_state = sbdr.ahtd.init_state_from_input(
        xs,
        **model,
    )
    z = sbdr.ahtd.extract_features(
        model_state,
        x_seq = xs,
        layer_idxs = layer_features,
        **model,
    )
    # Record temporal average of output
    zs_test.append(onp.array(z))

    # And categorical labels
    lab_cat = labels.argmax(axis=-1)
    labels_categorical_test.append(lab_cat)
    
# Convert to have numpy arrays

zs_test = onp.concatenate(zs_test, axis=0)
labels_categorical_test = onp.concatenate(labels_categorical_test, axis=0)

history_test = {
    "zs_val": zs_test,
    "labels_categorical_val": labels_categorical_test,
}

print(f"\tOutput shapes:")

pprint(sbdr.get_shapes(history_test))

print(f"\tAverage Activity: {history_test['zs_val'].mean()}")


"""---------------------"""
""" Save outputs """
"""---------------------"""

# save the activations to a compressed npz file
save_folder = os.path.join(
    model_folder,
    "activations",
)
os.makedirs(save_folder, exist_ok=True)

onp.savez_compressed(
    os.path.join(save_folder, f"activations_chkp_{default_step:03d}.npz"),
    **history,
    **history_test,
)