import os
import cv2
import jax
from jax import vmap, grad, jit
import jax.numpy as np
from functools import partial
import infomax_sbdr as sbdr
import optax
from pprint import pprint
import toml
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
from datetime import datetime

np.set_printoptions(precision=4, suppress=True)

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
    "-n", "--name", type=str, help="Name of model to train", default="test_1"
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
# add two dummy final dimensions, to make arguments broadcast compatible with the channel axis
kwargs_transform = {
    k: v.unsqueeze(-1).unsqueeze(-1) for k, v in kwargs_transform.items()
}
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
)

# import parameters and history if they exist
if os.path.exists(os.path.join(model_folder, f"params.pkl")):
    params, opt_state = sbdr.load_model(model_folder)
    PRE_LOADED = True
else:
    # # # Initialize parameters
    # Take some data
    xs, labels = next(iter(dataloader))
    # Generate key
    key = jax.random.key(model_config["model"]["seed"])
    # Init params
    params = model.init(key, xs)
    PRE_LOADED = False

print(f"\tDict of params: \n\t{params['params'].keys()}")


# print the shapes nicely
def get_shapes(nested_dict):
    return jax.tree_util.tree_map(lambda x: x.shape, nested_dict)


pprint(get_shapes(params["params"]))


"""---------------------"""
""" Forward Pass """
"""---------------------"""

print("\nForward pass jitted")


def forward_gen(params, xs, model):
    return model.apply(params, xs)


forward = jit(
    partial(
        forward_gen,
        model=model,
    ),
    static_argnames=("model",),
)

# test the forward pass
xs, labels = next(iter(dataloader))


zs, neg_pmi = forward(params, xs)

print(f"\tInput shape: {xs.shape}")
print(f"\tOutput Info:")
print(f"\t\tzs shape: {zs.shape}")
print(f"\t\tneg_pmi shape: {neg_pmi.shape}")
print(f"\t\tmean active units: {zs.sum(axis=-1).mean()}")
print(f"\t\tstd active units: {zs.sum(axis=-1).std()}")


"""---------------------"""
""" Loss Function """
"""---------------------"""

sim_fn = partial(
    sbdr.config_similarity_dict[model_config["training"]["loss"]["sim_fn"]["type"]],
    **model_config["training"]["loss"]["sim_fn"]["kwargs"],
)


def l2_weight_decay(w):
    return (w**2).mean()


def loss_fn_gen(
    params,
    xs,
    key,
    sim_fn,
    weight_decay,
    model,
):

    # # # Forward pass
    zs, neg_pmi = forward_gen(
        params,
        xs,
        model=model,
    )

    eps = 1.0e-6

    # # # Contrastive Mutual Information FLO estimator
    # # Positive samples
    p_ii_ctx = sim_fn(zs, zs)
    # # Negative samples
    p_ij_ctx = sim_fn(zs[..., :, None, :], zs[..., None, :, :])
    # # Neg-pmi term
    u_ii_ctx = neg_pmi[..., 0]
    # compute FLO estimator
    flo_loss = -sbdr.flo(u_ii_ctx, p_ii_ctx, p_ij_ctx, eps=eps)
    flo_loss = flo_loss.mean()

    # weight regularization
    weight_loss = sum(l2_weight_decay(w) for w in jax.tree.leaves(params["params"]))
    weight_loss = weight_loss.mean()
    weight_loss = weight_loss * weight_decay

    tot_loss = flo_loss + weight_loss

    return tot_loss, {
        "flo": flo_loss,
        "weight": weight_loss,
    }


loss_fn = partial(
    loss_fn_gen,
    sim_fn=sim_fn,
    model=model,
    **model_config["training"]["loss"]["kwargs"],
)

loss_fn = jit(loss_fn)

# test loss function

xs, labels = next(iter(dataloader))

key = jax.random.key(model_config["model"]["seed"])

loss, aux = loss_fn(params, xs, key)

print(f"\tLoss: {loss}")
print(f"\tAux: {aux}")


"""---------------------"""
""" Optimizer and params update """
"""---------------------"""

print("\nOptimizer and params update")

optimizer = sbdr.config_optimizer_dict[model_config["training"]["optimizer"]["type"]]
optimizer = optimizer(**model_config["training"]["optimizer"]["kwargs"])
if not PRE_LOADED:
    opt_state = optimizer.init(params)


# Init update function
def update_params_gen(
    params,
    opt_state,
    xs,
    key,
    sim_fn,
    weight_decay,
    model,
    optimizer,
):

    (loss, aux), grads = jax.value_and_grad(loss_fn_gen, 0, has_aux=True)(
        params,
        xs,
        key,
        sim_fn,
        weight_decay,
        model,
    )

    updates, opt_state = optimizer.update(grads, opt_state)

    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux, grads


update_params = partial(
    update_params_gen,
    sim_fn=sim_fn,
    model=model,
    optimizer=optimizer,
    **model_config["training"]["loss"]["kwargs"],
)
update_params = jit(update_params)

print("\t Test one update")

xs, labels = next(iter(dataloader))

key = jax.random.key(model_config["model"]["seed"])

tmp_params, tmp_opt_state, tmp_loss, tmp_aux, tmp_grads = update_params(
    params, opt_state, xs, key
)

print(f"\tLoss: {tmp_loss}")

gradients_are_finite = jax.tree_util.tree_map(
    lambda x: np.all(np.isfinite(x)).item(), tmp_grads
)
gradients_are_zero = jax.tree_util.tree_map(
    lambda x: np.all(np.isclose(x, 0.0, rtol=1e-20, atol=1e-20)).item(), tmp_grads
)
gradients_are_nan = jax.tree_util.tree_map(
    lambda x: np.any(np.isnan(x)).item(), tmp_grads
)
print(f"\tGradients are FINITE:")
pprint(gradients_are_finite)
print(f"\tGradients are ZERO (tolerance):")
pprint(gradients_are_zero)
print(f"\tGradients are NAN:")
pprint(gradients_are_nan)


"""------------------"""
""" Training """
"""------------------"""

# if not PRE_LOADED:
# initialize history
history = {
    "loss/train": [],
    "aux/flo": [],
    "aux/weight": [],
}


key = jax.random.key(42)


writer = SummaryWriter(
    log_dir=os.path.join(
        model_folder,
        "logs",
        f"{datetime.today().strftime('%Y%m%d-%H%M%S')}",
    )
)

log_interval = model_config["training"]["log_interval"]
save_interval = model_config["training"]["save_interval"]
save_model = model_config["training"]["save"]
key = jax.random.key(model_config["model"]["seed"])

print("\nTraining")

# history that collect the loss over the batches of a single epoch
history_epoch = {k: [] for k in history.keys()}
# history that collects data for log_interval epochs
# then is printed and reset
history_interval = {k: [] for k in history.keys()}

try:

    for epoch in range(model_config["training"]["epochs"]):

        # history that collect the loss over the batches of a single epoch
        history_epoch = {k: [] for k in history.keys()}

        for k in history_interval.keys():
            history_interval[k].append([])

        for batch_idx, (xs, labels) in enumerate(dataloader):

            # random flip of images on width dimension for data augmentation
            key, _ = jax.random.split(key)
            mask = jax.random.bernoulli(key, p=0.5, shape=(xs.shape[0], 1, 1, 1))
            # flip on last axis, (C,H,W) format is assumed
            xs = xs * mask + (1 - mask) * xs[..., ::-1]

            # applu a bit of gaussian noise
            key, _ = jax.random.split(key)
            xs = np.clip(xs + 0.1 * jax.random.normal(key, shape=xs.shape), 0, 1)

            key, _ = jax.random.split(key)
            params, opt_state, loss, aux, grads = update_params(
                params, opt_state, xs, key
            )

            history_epoch["loss/train"].append(loss.item())
            history_epoch["aux/flo"].append(aux["flo"].item())
            history_epoch["aux/weight"].append(aux["weight"].item())

            history_interval["loss/train"][-1].append(loss.item())
            history_interval["aux/flo"][-1].append(aux["flo"].item())
            history_interval["aux/weight"][-1].append(aux["weight"].item())

        history["loss/train"].append(
            (np.array(history_epoch["loss/train"]).mean()).item()
        )
        history["aux/flo"].append((np.array(history_epoch["aux/flo"]).mean()).item())
        history["aux/weight"].append(
            (np.array(history_epoch["aux/weight"]).mean()).item()
        )

        # log to the writer
        for k, v in history.items():
            writer.add_scalar(k, v[-1], epoch)

        # Logging
        if epoch % log_interval == 0:
            history_interval = {
                k: np.array(v).mean(axis=-1) for k, v in history_interval.items()
            }
            # Print useful information
            print(f"Epoch {epoch}:")
            for k, v in history_interval.items():
                print(f"\t{k}: {v}")

            # reset the history of the log interval
            history_interval = {k: [] for k in history.keys()}

        # Save model
        if (epoch % save_interval == 0) and (epoch != 0) and save_model:
            sbdr.save_model(params, history, opt_state, model_folder)

except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()

if save_model:
    sbdr.save_model(params, history, opt_state, model_folder)
