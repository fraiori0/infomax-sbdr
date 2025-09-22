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
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

# np.set_printoptions(precision=4, suppress=True)

BINARIZE = True  # whether to binarize the outputs or not
BINARIZE_THRESHOLD = 0.5  # threshold for binarization, only used if BINARIZE is True

default_model = "vgg_sigmoid_logand"  # "vgg_sbdr_5softmax/1"  #
default_number = "2"
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

# print(f"\nTest time for one train epoch:")
# # test time for one epoch
# t0 = time()
# for xs, labels in tqdm(dataloader_train):
#     forward_eval_jitted(variables, xs)

# print(f"\tTime for one epoch: {time() - t0}")

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
""" Import MLP config and create dataloaders for classifier"""
"""---------------------"""

class_model_folder = os.path.join(
    model_folder,
    "mlp_classifier",
)

print("\nImport MLP config and create dataloaders for classifier")

# import the elman_config.toml
with open(os.path.join(class_model_folder, "config.toml"), "r") as f:
    class_model_config = toml.load(f)


class_dataset_train = sbdr.ClassifierDataset(
    x = onp.array(zs),
    categorical_labels=onp.array(labels_categorical),
)
class_dataset_val = sbdr.ClassifierDataset(
    x = onp.array(zs_val),
    categorical_labels=onp.array(labels_categorical_val),
)


class_dataloader_train = sbdr.NumpyLoader(
    class_dataset_train,
    batch_size=class_model_config["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
)

class_dataloader_val = sbdr.NumpyLoader(
    class_dataset_val,
    batch_size=256,
    shuffle=False,
    drop_last=False,
)

"""---------------------"""
""" Init Classifier """
"""---------------------"""

print("\nInit Classifier")

class_model_class = sbdr.config_classifier_module_dict[class_model_config["model"]["type"]]

class_model = class_model_class(
    **class_model_config["model"]["kwargs"],
    activation_fn=sbdr.config_activation_dict[class_model_config["model"]["activation"]],
)


# # # Initialize parameters
# Take some data
xs, labels = next(iter(class_dataloader_train))
# Generate key
key = jax.random.key(class_model_config["model"]["seed"])
# Init params and batch_stats
class_variables = class_model.init(key, xs)


print(f"\tDict of variables: \n\t{class_variables.keys()}")


@jit
def class_forward(variables, xs):
    return class_model.apply(variables, xs)

# Perform a forward pass just to test and print shapes

outs = class_forward(class_variables, xs)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))

"""---------------------"""
""" Loss Function """
"""---------------------"""

def crossentropy_loss(outs, onehot_labels):
    # Compute categorical crossentropy loss
    logits = outs["logits"]
    loss = -(onehot_labels * jax.nn.log_softmax(logits)).sum(axis=-1)
    return loss.mean()

crossentropy_loss_jitted = jit(crossentropy_loss)

def accuracy(outs, onehot_labels):
    # Compute accuracy
    preds = outs["logits"].argmax(axis=-1)
    acc = (preds == onehot_labels.argmax(axis=-1)).mean()
    return acc

accuracy_jitted = jit(accuracy)


# test loss function

loss = crossentropy_loss_jitted(outs, labels)
print(f"\tLoss: {loss}")

acc = accuracy_jitted(outs, labels)
print(f"\tAccuracy: {acc}")



"""---------------------"""
""" Checkpointing """
"""---------------------"""


class_checkpoint_manager = orbax.checkpoint.CheckpointManager(
    directory=os.path.join(class_model_folder, "checkpoints"),
    # checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
    options=orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=class_model_config["training"]["checkpoint"]["save_interval"],
        max_to_keep=class_model_config["training"]["checkpoint"]["max_to_keep"],
        step_format_fixed_length=5,
    ),
)

print("\nCheckpointing stuff")

class_optimizer = sbdr.config_optimizer_dict[class_model_config["training"]["optimizer"]["type"]]
class_optimizer = class_optimizer(**class_model_config["training"]["optimizer"]["kwargs"])

class_state = {
    "variables": class_variables,
    "opt_state": class_optimizer.init(class_variables["params"]),
    "step": 0,
}

LAST_STEP = class_checkpoint_manager.latest_step()
if LAST_STEP is None:
    LAST_STEP = 0
else:
    class_state = class_checkpoint_manager.restore(
        step=LAST_STEP, args=orbax.checkpoint.args.StandardRestore(class_state)
    )

print(f"\tLast step: {LAST_STEP}")


"""---------------------"""
""" Training and Evaluation Steps """
"""---------------------"""

print("\nTraining and Evaluation Steps")

@jit
def class_train_step(class_state, batch):

    def loss_fn(params):
        # Apply the model
        outs = class_forward(
            {"params": params},
            batch["x"],
        )
        # Compute loss
        loss_val = crossentropy_loss_jitted(outs, batch["label"])
        # compute accuracy
        acc_val = accuracy(outs, batch["label"])
        metrics = {
            "loss/crossentropy": loss_val,
            "loss/accuracy": acc_val,
        }
        return loss_val, metrics

    # compute gradient, loss, and aux
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss_val, metrics), grads = grad_fn(class_state["variables"]["params"])
    # update weights
    updates, opt_state = class_optimizer.update(grads, class_state["opt_state"], class_state["variables"]["params"])
    class_state["variables"]["params"] = optax.apply_updates(
        class_state["variables"]["params"], updates
    )
    # Update optimizer state
    class_state["opt_state"] = opt_state

    # update step
    class_state["step"] += 1
    return class_state, metrics, grads, 

@jit
def class_eval_step(class_state, batch):

    def loss_fn(params):
        # Apply the model
        outs = class_forward(
            {"params": params},
            batch["x"],
        )
        # Compute loss
        loss_val = crossentropy_loss_jitted(outs, batch["label"])
        # compute accuracy
        acc_val = accuracy(outs, batch["label"])
        metrics = {
            "loss/crossentropy": loss_val,
            "loss/accuracy": acc_val,
        }
        return loss_val, metrics

    loss_val, metrics = loss_fn(class_state["variables"]["params"])

    return metrics

print("\tTest one train step and one eval step")

xs, labels = next(iter(class_dataloader_train))
key = jax.random.key(class_model_config["model"]["seed"])
batch = {
    "x": xs,
    "label": labels,
    "key": key,
}

class_state, metrics, grads = class_train_step(class_state, batch)
print(f"\tLoss: {metrics['loss/crossentropy']}")
print(f"\tAccuracy: {metrics['loss/accuracy']}")
class_state, metrics, grads = class_train_step(class_state, batch)

# with jax.checking_leaks():
#     metrics_val = class_eval_step(class_state, batch)



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

key = jax.random.key(class_model_config["model"]["seed"])

# tensorboard writer
class_log_folder = os.path.join(
    class_model_folder,
    "logs",
    datetime.now().strftime("%Y%m%d_%H%M%S"),
)
writer = SummaryWriter(log_dir=class_log_folder)

try:
    for epoch in tqdm(range(class_model_config["training"]["epochs"])):
        for batch_idx, (xs, labels) in tqdm(enumerate(class_dataloader_train), leave=False, total=len(class_dataloader_train)):
            key, _ = jax.random.split(key)
            batch = {
                "x": xs,
                "label": labels,
                "key": key,
            }
            class_state, metrics, grads = class_train_step(class_state, batch)
            metrics = jax.tree_util.tree_map(lambda x: x.item(), metrics)
            writer.add_scalar("train/loss", metrics["loss/crossentropy"], class_state["step"].item())
            writer.add_scalar("train/accuracy", metrics["loss/accuracy"], class_state["step"].item())

            # print(class_state["step"].item())

        # test on validation set
        for eval_step, (xs, labels) in tqdm(enumerate(class_dataloader_val), leave=False, total=len(class_dataloader_val)):
            key, _ = jax.random.split(key)
            batch = {
                "x": xs,
                "label": labels,
                "key": key,
            }
            metrics = class_eval_step(class_state, batch)
            metrics = jax.tree_util.tree_map(lambda x: x.item(), metrics)
            writer.add_scalar("val/loss", metrics["loss/crossentropy"], eval_step + len(class_dataloader_train) * epoch)
            writer.add_scalar("val/accuracy", metrics["loss/accuracy"], eval_step + len(class_dataloader_train) * epoch)

        # Save checkpoint
        if class_model_config["training"]["checkpoint"]["save"]:
            # save checkpoint
            class_checkpoint_manager.save(
                step=int(class_state["step"].item() / (len(class_dataloader_train))),
                args=orbax.checkpoint.args.StandardSave(class_state),
            )
except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()