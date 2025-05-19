import os
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

import infomax_sbdr as sbdr

from torch.utils.tensorboard import SummaryWriter
import torch


np.set_printoptions(precision=4, suppress=True)

default_model = "vgg_sbdr"

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
    "-n",
    "--name",
    type=str,
    help="Name of model to train",
    default=default_model,
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
    args.name,
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

kwargs_transform = model_config["dataset"]["transform"]["kwargs"]
# convert all kwargs to torch arrays (the transform is applied to the torch arrays in the dataset class, before they are converted to jax numpy arrays)
kwargs_transform = {k: torch.tensor(v) for k, v in kwargs_transform.items()}

# create the transform
transform = partial(
    sbdr.config_transform_dict[model_config["dataset"]["transform"]["type"]],
    **kwargs_transform,
)

dataset = sbdr.Cifar10Dataset(
    folder_path=data_folder,
    kind="train",
    transform=transform,
    flatten=model_config["dataset"]["flatten"],
)

dataset_train, dataset_val = torch.utils.data.random_split(
    dataset,
    [
        1 - model_config["training"]["early_stopping"]["val_split"],
        model_config["training"]["early_stopping"]["val_split"],
    ],
    generator=torch.Generator().manual_seed(model_config["model"]["seed"]),
)


dataloader_train = sbdr.NumpyLoader(
    dataset_train,
    batch_size=model_config["training"]["batch_size"],
    shuffle=model_config["training"]["shuffle"],
    drop_last=model_config["training"]["drop_last"],
)

dataloader_val = sbdr.NumpyLoader(
    dataset_val,
    batch_size=2 * model_config["training"]["batch_size"],
    shuffle=False,
    drop_last=model_config["training"]["drop_last"],
)

xs, labels = next(iter(dataloader_train))

print(f"\tInput xs: {xs.shape} (dtype: {xs.dtype})")
print(f"\tLabels: {labels.shape} (dtype: {labels.dtype})")


"""---------------------"""
""" Init Network """
"""---------------------"""

print("\nInitializing model")

model_class = sbdr.config_module_dict[model_config["model"]["type"]]

model = model_class(
    **model_config["model"]["kwargs"],
    activation_fn=sbdr.config_activation_dict[model_config["model"]["activation"]],
    out_activation_fn=sbdr.config_activation_dict[
        model_config["model"]["out_activation"]
    ],
    training=True,
)

model_eval = model_class(
    **model_config["model"]["kwargs"],
    activation_fn=sbdr.config_activation_dict[model_config["model"]["activation"]],
    out_activation_fn=sbdr.config_activation_dict[
        model_config["model"]["out_activation"]
    ],
    training=False,
)

checkpoint_manager = orbax.checkpoint.CheckpointManager(
    directory=os.path.join(model_folder, "checkpoints"),
    checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
    options=orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=model_config["training"]["checkpoint"]["save_interval"],
        max_to_keep=model_config["training"]["checkpoint"]["max_to_keep"],
        step_format_fixed_length=5,
    ),
)

# import previous save, if it exists
LOADED = False
LAST_STEP = 0
try:
    LAST_STEP = checkpoint_manager.latest_step()
    ckpt = checkpoint_manager.restore(step=LAST_STEP)
    state = ckpt["state"]
    print(state.keys())
    variables = {
        "params": state["variables"]["params"],
        "batch_stats": state["variables"]["batch_stats"],
    }
    LOADED = True
    print("\nLoaded previous checkpoint")
except FileNotFoundError:
    print("\nNo previous checkpoint found. Initializing parameters")
    # # # Initialize parameters
    # Take some data
    xs, labels = next(iter(dataloader_train))
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


def forward(variables, xs, key):
    return model.apply(
        variables,
        xs,
        # key for dropout
        rngs={"dropout": key},
        # batch stats should be updated
        mutable=["batch_stats"],
    )


def forward_eval(variables, xs):
    return model_eval.apply(
        variables,
        xs,
        mutable=["batch_stats"],
    )


forward_jitted = jit(forward)
forward_eval_jitted = jit(forward_eval)

# test the forward pass
xs, labels = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])

(zs, neg_pmi), mutable_updates = forward_jitted(variables, xs, key)
(zs, neg_pmi), _ = forward_eval_jitted(variables, xs)


print(f"\tInput shape: {xs.shape}")
print(f"\tOutput Info:")
print(f"\t\tzs shape: {zs.shape}")
print(f"\t\tneg_pmi shape: {neg_pmi.shape}")
print(f"\t\tmean active units: {zs.sum(axis=-1).mean()}")
print(f"\t\tstd active units: {zs.sum(axis=-1).std()}")


pprint(get_shapes(mutable_updates))

# test time for one epoch
t0 = time()
for xs, labels in dataloader_train:
    key, _ = jax.random.split(key)
    forward_jitted(variables, xs, key)

print(f"\tTime for one epoch: {time() - t0}")

"""---------------------"""
""" Loss Function """
"""---------------------"""

sim_fn = partial(
    sbdr.config_similarity_dict[model_config["training"]["loss"]["sim_fn"]["type"]],
    **model_config["training"]["loss"]["sim_fn"]["kwargs"],
)


def l2_weight_loss(w):
    return (w**2).mean()


def flo_loss(
    zs,
    negpmi,
):

    eps = 1.0e-6

    # # # Contrastive Mutual Information FLO estimator
    # # Positive samples
    p_ii_ctx = sim_fn(zs, zs)
    # # Negative samples
    p_ij_ctx = sim_fn(zs[..., :, None, :], zs[..., None, :, :])
    # # Neg-pmi term
    u_ii_ctx = negpmi[..., 0]
    # compute FLO estimator
    flo_loss = -sbdr.flo(u_ii_ctx, p_ii_ctx, p_ij_ctx, eps=eps)
    flo_loss = flo_loss.mean()

    return flo_loss


flo_loss_jitted = jit(flo_loss)

# test loss function
xs, labels = next(iter(dataloader_train))

key = jax.random.key(model_config["model"]["seed"])

(zs, negpmi), _ = forward(variables, xs, key)

print(zs.shape)
print(negpmi.shape)

loss = flo_loss_jitted(zs, negpmi)

print(f"\tFLO Loss: {loss}")


"""---------------------"""
""" Optimizer and Train State """
"""---------------------"""

print("\nOptimizer and Train State")


# add batch stats to train state
optimizer = sbdr.config_optimizer_dict[model_config["training"]["optimizer"]["type"]]
optimizer = optimizer(**model_config["training"]["optimizer"]["kwargs"])

if not LOADED:
    state = {
        "variables": variables,
        "opt_state": optimizer.init(variables["params"]),
        "step": 0,
    }


@jit
def train_step(state, batch):
    def loss_fn(params):
        (zs, negpmi), mutable_updates = forward(
            {
                "params": params,
                "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x"],
            batch["key"],
        )
        flo_loss_val = flo_loss_jitted(zs, negpmi)
        # weight_loss_val = l2_weight_loss(params["params"])
        loss = flo_loss_val  # + weight_loss_val
        return loss, {
            "flo_loss": flo_loss_val,
            "weight_loss": 0.0,
            "mutable_updates": mutable_updates,
        }

    # compute gradient, loss, and aux
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, aux), grads = grad_fn(state["variables"]["params"])
    # update weights
    updates, opt_state = optimizer.update(grads, state["opt_state"])
    state["variables"]["params"] = optax.apply_updates(
        state["variables"]["params"], updates
    )
    state["opt_state"] = opt_state
    # update batch stats
    state["variables"]["batch_stats"] = aux["mutable_updates"]["batch_stats"]
    # update step
    state["step"] += 1

    # Store metrics
    metrics = {
        "loss/total": loss,
        "loss/flo": aux["flo_loss"],
        "loss/weight": aux["weight_loss"],
    }

    return state, metrics, grads


@jit
def eval_step(state, batch):
    def loss_fn(variables):
        (zs, negpmi), mutable_updates = forward_eval(
            variables,
            batch["x"],
        )
        flo_loss_val = flo_loss_jitted(zs, negpmi)
        # weight_loss_val = l2_weight_loss(params["params"])
        loss = flo_loss_val  # + weight_loss_val
        return loss, {
            "flo_loss": flo_loss_val,
            "weight_loss": 0.0,
            "mutable_updates": mutable_updates,
        }

    loss, aux = loss_fn(
        variables=state["variables"],
    )
    metrics = {
        "loss/total": loss,
        "loss/flo": aux["flo_loss"],
        "loss/weight": aux["weight_loss"],
    }

    return state, metrics


print("\t Test one train step and one eval step")

xs, labels = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])
batch = {
    "x": xs,
    "key": key,
}

state, metrics, grads = train_step(state, batch)
_, _ = eval_step(state, batch)

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

    for epoch in range(model_config["training"]["epochs"]):

        epoch_metrics = {k: 0.0 for k in metrics.keys()}

        for batch_idx, (xs, labels) in enumerate(dataloader_train):

            # random flip of images on width dimension for data augmentation
            key, _ = jax.random.split(key)
            mask = jax.random.bernoulli(key, p=0.5, shape=(xs.shape[0], 1, 1, 1))
            # flip on second-to-last last axis, (H,W,C) format is assumed
            xs = xs * mask + (1 - mask) * xs[..., ::-1, :]

            key, _ = jax.random.split(key)
            batch = {
                "x": xs,
                "key": key,
            }

            state, metrics, grads = train_step(state, batch)

            LAST_STEP = state["step"].item()

            for metric, value in metrics.items():
                writer.add_scalar(
                    metric + "/train/batch", value.item(), global_step=LAST_STEP
                )

            for metric, value in metrics.items():
                epoch_metrics[metric] += value

        # take the average of the epoch stats
        epoch_metrics = jax.tree.map(lambda x: x / (batch_idx + 1), epoch_metrics)
        epoch_step = int(LAST_STEP / (batch_idx + 1))

        # Log epoch stats
        for metric, value in epoch_metrics.items():
            writer.add_scalar(
                metric + "/train_epoch", value.item(), global_step=epoch_step
            )

        print(f"Epoch {epoch}/{model_config['training']['epochs']}:")
        pprint(epoch_metrics)

        # COmpute average loss on validation set
        val_metrics = {k: 0.0 for k in metrics.keys()}
        for val_batch_idx, (xs, labels) in enumerate(dataloader_val):
            batch = {
                "x": xs,
            }

            state, val_metrics = eval_step(state, batch)

            for metric, value in val_metrics.items():
                val_metrics[metric] += value

        # take the average of the epoch stats
        val_metrics = jax.tree.map(lambda x: x / (val_batch_idx + 1), val_metrics)

        # Log epoch stats
        for metric, value in val_metrics.items():
            writer.add_scalar(
                metric + "/val_epoch", value.item(), global_step=epoch_step
            )

        print(f"Validation:")
        pprint(val_metrics)

        if (
            (epoch > 0)
            and (epoch % model_config["training"]["save_interval"] == 0)
            and model_config["training"]["save"]
        ):
            # save checkpoint
            checkpoint_manager.save(
                step=LAST_STEP,
                items={
                    "state": state,
                },
            )


except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()
