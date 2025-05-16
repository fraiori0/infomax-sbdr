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

dataset_train = sbdr.Cifar10Dataset(
    folder_path=data_folder,
    kind="train",
    transform=transform,
    flatten=model_config["dataset"]["flatten"],
)

dataloader = sbdr.NumpyLoader(
    dataset_train,
    batch_size=model_config["training"]["batch_size"],
    shuffle=model_config["training"]["shuffle"],
    drop_last=model_config["training"]["drop_last"],
)

xs, labels = next(iter(dataloader))

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
    variables = {
        "params": ckpt["params"],
        "batch_stats": ckpt["batch_stats"],
    }
    LOADED = True
    print("\nLoaded previous checkpoint")
except FileNotFoundError:
    print("\nNo previous checkpoint found. Initializing parameters")
    # # # Initialize parameters
    # Take some data
    xs, labels = next(iter(dataloader))
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


def forward(params, xs, key):
    return model.apply(
        {"params": params},
        xs,
        # key for dropout
        rngs={"dropout": key},
        # batch stats should be updated
        mutable=["batch_stats"],
    )


forward_jitted = jit(forward)

# test the forward pass
xs, labels = next(iter(dataloader))
key = jax.random.key(model_config["model"]["seed"])

(zs, neg_pmi), mutable_updates = forward_jitted(variables["params"], xs, key)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput Info:")
print(f"\t\tzs shape: {zs.shape}")
print(f"\t\tneg_pmi shape: {neg_pmi.shape}")
print(f"\t\tmean active units: {zs.sum(axis=-1).mean()}")
print(f"\t\tstd active units: {zs.sum(axis=-1).std()}")

pprint(get_shapes(mutable_updates))

# test time for one epoch
t0 = time()
for xs, labels in dataloader:
    key, _ = jax.random.split(key)
    forward_jitted(variables["params"], xs, key)

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
xs, labels = next(iter(dataloader))

key = jax.random.key(model_config["model"]["seed"])

(zs, negpmi), _ = forward(variables["params"], xs, key)

print(zs.shape)
print(negpmi.shape)

loss = flo_loss_jitted(zs, negpmi)

print(f"\tFLO Loss: {loss}")


"""---------------------"""
""" Optimizer and Train State """
"""---------------------"""

print("\nOptimizer and Train State")


# add batch stats to train state
class TrainState(train_state.TrainState):
    batch_stats: Any


if not LOADED:
    optimizer = sbdr.config_optimizer_dict[
        model_config["training"]["optimizer"]["type"]
    ]
    optimizer = optimizer(**model_config["training"]["optimizer"]["kwargs"])
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=variables["batch_stats"],
    )


@jit
def train_step(state, batch):
    def loss_fn(params):
        (zs, negpmi), mutable_updates = forward(
            params,
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

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)

    # update weights
    state = state.apply_gradients(grads=grads)

    state = state.replace(batch_stats=aux["mutable_updates"]["batch_stats"])

    metrics = {
        "loss/total": loss,
        "loss/flo": aux["flo_loss"],
        "loss/weight": aux["weight_loss"],
    }

    aux = grads

    return state, metrics, grads


print("\t Test one train step")

xs, labels = next(iter(dataloader))
key = jax.random.key(model_config["model"]["seed"])
batch = {
    "x": xs,
    "key": key,
}

state, metrics, grads = train_step(state, batch)

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

        epoch_stats = {k: 0.0 for k in metrics.keys()}

        for batch_idx, (xs, labels) in enumerate(dataloader):

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

            for metric, value in metrics.items():
                writer.add_scalar(metric, value.item(), global_step=state.step)

            for metric, value in metrics.items():
                epoch_stats[metric] += value

        # take the average of the epoch stats
        epoch_stats = jax.tree.map(lambda x: x / (batch_idx + 1), epoch_stats)

        print(f"Epoch {epoch}/{model_config['training']['epochs']}:")
        pprint(epoch_stats)

        if (
            (epoch > 0)
            and (epoch % model_config["training"]["save_interval"] == 0)
            and (epoch % model_config["training"]["save"])
        ):
            # TODO: here is never executed
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            # save checkpoint
            checkpoint_manager.save(
                step=state.step,
                items={
                    "params": state.params,
                    "batch_stats": state.batch_stats,
                },
            )


except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()
