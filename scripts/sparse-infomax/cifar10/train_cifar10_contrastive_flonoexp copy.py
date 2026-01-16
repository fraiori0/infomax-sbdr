import os
import argparse



default_model = "tmp"
default_number = "1"

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
    default=default_number,
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
        # random resize and cropping
        tv_transforms.RandomResizedCrop(
            size=model_config["dataset"]["transform"]["resized_crop"]["size"],
            scale=model_config["dataset"]["transform"]["resized_crop"]["scale"],
            ratio=model_config["dataset"]["transform"]["resized_crop"]["ratio"],
        ),
        # random horizontal flip
        tv_transforms.RandomHorizontalFlip(
            p=model_config["dataset"]["transform"]["flip"]["p"],
        ),
        # random color jitter
        tv_transforms.ColorJitter(
            brightness=model_config["dataset"]["transform"]["color_jitter"][
                "brightness"
            ],
            contrast=model_config["dataset"]["transform"]["color_jitter"]["contrast"],
            saturation=model_config["dataset"]["transform"]["color_jitter"][
                "saturation"
            ],
            hue=model_config["dataset"]["transform"]["color_jitter"]["hue"],
        ),
        # random grayscale
        tv_transforms.RandomGrayscale(
            p=model_config["dataset"]["transform"]["grayscale"]["p"],
        ),
        # normalize
        tv_transforms.Normalize(
            mean=model_config["dataset"]["transform"]["normalization"]["mean"],
            std=model_config["dataset"]["transform"]["normalization"]["std"],
        ),
        # change from  (C, H, W) to (H, W, C)
        tv_transforms.Lambda(lambda x: x.movedim(-3, -1)),
    ]
)

dataset = sbdr.Cifar10DatasetContrastive(
    folder_path=data_folder,
    kind="train",
    transform=transform,
    device="cpu", #torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
    num_workers=2,
)

dataloader_val = sbdr.NumpyLoader(
    dataset_val,
    batch_size=model_config["validation"]["dataloader"]["batch_size"],
    shuffle=False,
    drop_last=True,
    # num_workers=2,
)

for (xs_1, xs_2), labels in dataloader_train:
    xs_1 = jax.device_put(xs_1)
    xs_2 = jax.device_put(xs_2)
    labels = jax.device_put(labels)
    break
# (xs_1, xs_2), labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)

print(f"\tInput xs: {xs_1.shape} (dtype: {xs_1.dtype})")
print(f"\tLabels: {labels.shape} (dtype: {labels.dtype})")

# Create a dataset with the original images
transform_original = tv_transforms.Compose(
    [
        tv_transforms.Normalize(
            mean=model_config["dataset"]["transform"]["normalization"]["mean"],
            std=model_config["dataset"]["transform"]["normalization"]["std"],
        ),
        # change from  (C, H, W) to (H, W, C)
        tv_transforms.Lambda(lambda x: x.movedim(-3, -1)),
    ]
)
dataset_original = sbdr.Cifar10Dataset(
    folder_path=data_folder,
    kind="train",
    transform=transform_original,
)
# use same split and keep only the originals from the train set
dataset_original, _ = torch.utils.data.random_split(
    dataset_original,
    [
        1 - model_config["validation"]["split"],
        model_config["validation"]["split"],
    ],
    generator=torch.Generator().manual_seed(model_config["model"]["seed"]),
)

dataloader_original = sbdr.NumpyLoader(
    dataset_original,
    batch_size=16,
    shuffle=model_config["training"]["dataloader"]["shuffle"],
    drop_last=model_config["training"]["dataloader"]["drop_last"],
)


"""---------------------"""
""" Visualize the images"""
"""---------------------"""


# def untransform(x):
#     return (x * np.array([0.2470, 0.2435, 0.2616])) + np.array([0.4914, 0.4822, 0.4465])


# # visualize the images
# # converted to BGR
# for i, (img_1, img_2) in enumerate(zip(xs_1, xs_2)):
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
#     # show image
#     cv2.imshow("Image 1", img_1)
#     cv2.imshow("Image 2", img_2)
#     k = cv2.waitKey(0)
#     if k == ord("q"):
#         break
# cv2.destroyAllWindows()

# exit()
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

# # # Initialize parameters
# Take some data
(xs_1, xs_2), labels = next(iter(dataloader_train))
xs_1 = jax.device_put(xs_1)
xs_2 = jax.device_put(xs_2)
# Generate key
key = jax.random.key(model_config["model"]["seed"])
# Init params and batch_stats
variables = model.init(key, xs_1)

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
        # DROPOUT
        # key for dropout
        rngs={"dropout": key},
        # BATCH_NORM
        # batch stats should be updated
        mutable=["batch_stats"],
    )


def forward_eval(variables, xs):
    return model_eval.apply(
        variables,
        xs,
        # BATCH_NORM
        mutable=["batch_stats"],
    )


forward_jitted = jit(forward)
forward_eval_jitted = jit(forward_eval)

# test the forward pass
(xs_1, xs_2), labels = next(iter(dataloader_train))
xs_1 = jax.device_put(xs_1)
xs_2 = jax.device_put(xs_2)
key = jax.random.key(model_config["model"]["seed"])

outs, mutable_updates = forward_jitted(variables, xs_1, key)
_, _ = forward_eval_jitted(variables, xs_1)

print(f"\tInput shape: {xs_1.shape}")
print(f"\tOutput shapes:")
pprint(get_shapes(outs))

print(f"\tMutable updates:")
pprint(get_shapes(mutable_updates))

# print(f"\nTest one epoch:")
# # test time for one epoch
# t0 = time()
# for (xs_1, xs_2), labels in tqdm(dataloader_train):
#     key, _ = jax.random.split(key)
#     forward_jitted(variables, xs_1, key)

# print(f"\tTime for one epoch: {time() - t0}")
exit()

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
    outs_1,
    outs_2,
):

    eps = 1.0e-6

    # # # Contrastive Mutual Information FLO estimator
    # # Positive samples
    p_ii_ctx = sim_fn(outs_1["z"], outs_2["z"])
    # # Negative samples
    p_ij_ctx_1 = sim_fn(outs_1["z"][..., :, None, :], outs_1["z"][..., None, :, :])
    p_ij_ctx_2 = sim_fn(outs_2["z"][..., :, None, :], outs_2["z"][..., None, :, :])
    # # Neg-pmi term
    u_ii_ctx_1 = outs_1["neg_pmi"][..., 0]
    u_ii_ctx_2 = outs_2["neg_pmi"][..., 0]
    # compute FLO estimator
    flo_loss_1 = -sbdr.flo(u_ii_ctx_1, p_ii_ctx, p_ij_ctx_1, eps=eps)
    flo_loss_2 = -sbdr.flo(u_ii_ctx_2, p_ii_ctx, p_ij_ctx_2, eps=eps)
    flo_loss = (flo_loss_1 + flo_loss_2) / 2
    flo_loss = flo_loss.mean()

    return flo_loss


flo_loss_jitted = jit(flo_loss)

# test loss function
(xs_1, xs_2), labels = next(iter(dataloader_train))

key = jax.random.key(model_config["model"]["seed"])
key_1, key_2 = jax.random.split(key)

outs_1, _ = forward(variables, xs_1, key_1)
outs_2, _ = forward(variables, xs_2, key_2)

loss = flo_loss_jitted(outs_1, outs_2)

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
""" Training and Evaluation Steps """
"""---------------------"""

print("\nTraining and Evaluation Steps")


@jit
def train_step(state, batch):
    def loss_fn(params):
        # Apply the model
        outs_1, mutable_updates_1 = forward(
            {
                "params": params,
                # BATCH_NORM - change here
                "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x_1"],
            batch["key_1"],
        )
        # Apply the model to the augmented images
        outs_2, mutable_updates_2 = forward(
            {
                "params": params,
                # BATCH_NORM - change here
                "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x_2"],
            batch["key_2"],
        )
        
        # # ADD NOISE - change here
        # key1_p, key1_n = jax.random.split(batch["key_1"])
        # key2_p, key2_n = jax.random.split(batch["key_2"])
        # noise_p = 0.005
        # noise_n = 0.02
        # mask_p_1 = jax.random.bernoulli(key1_p, p=1-noise_p, shape=outs_1["z"].shape).astype(outs_1["z"].dtype)
        # mask_p_2 = jax.random.bernoulli(key2_p, p=1-noise_p, shape=outs_2["z"].shape).astype(outs_2["z"].dtype)
        # mask_n_1 = jax.random.bernoulli(key1_n, p=1-noise_n, shape=outs_1["z"].shape).astype(outs_1["z"].dtype)
        # mask_n_2 = jax.random.bernoulli(key2_n, p=1-noise_n, shape=outs_2["z"].shape).astype(outs_2["z"].dtype)
        # # OR with the positive noise, AND with the negative noise
        # outs_1["z"] = (1-((1-outs_1["z"]) * mask_p_1)) * mask_n_1
        # outs_2["z"] = (1-((1-outs_2["z"]) * mask_p_2)) * mask_n_2
        

        # Compute FLO loss
        flo_loss_val = flo_loss(outs_1, outs_2)
        loss_val = flo_loss_val

        # # Compute reconstruction loss
        # rec_loss_val = ((outs["xs_rec"] - batch["x"]) ** 2).mean()
        # rec_loss_val = rec_loss_val * model_config["training"]["loss"]["mse_scale"]
        # alpha = model_config["training"]["loss"]["alpha"]
        # loss_val = alpha * rec_loss_val + (1 - alpha) * loss_val

        # # Compute L1 norm of the output
        # l1_norm_1 = np.abs(outs_1["z"]).mean()
        # l1_norm_2 = np.abs(outs_2["z"]).mean()
        # l1_norm_val = (l1_norm_1 + l1_norm_2) / 2
        # loss_val = loss_val + model_config["training"]["loss"]["l1_norm"]["scale"] * l1_norm_val

        # # Compute sparsity
        sparsity_val = (outs_1["z"].mean() + outs_2["z"].mean()) / 2

        metrics = {
            "loss/total": loss_val,
            "loss/flo": flo_loss_val,
            # "loss/rec": rec_loss_val,
            "sparsity": sparsity_val,
        }

        others = {
            "mutable_updates": jax.tree.map(
                lambda x, y: (x + y) / 2.0, mutable_updates_1, mutable_updates_2
            ),
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

    # BATCH_NORM - change here
    # update batch stats
    state["variables"]["batch_stats"] = others["mutable_updates"]["batch_stats"]

    # update step
    state["step"] += 1

    return state, metrics, grads


@jit
def eval_step(state, batch):
    def loss_fn(params):
        # Apply the model
        outs_1, mutable_updates_1 = forward_eval(
            {
                "params": params,
                # BATCH_NORM - change here
                "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x_1"],
        )
        # Apply the model to the augmented images
        outs_2, mutable_updates_2 = forward_eval(
            {
                "params": params,
                # BATCH_NORM - change here
                "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x_2"],
        )
        # Compute FLO loss
        flo_loss_val = flo_loss(outs_1, outs_2)
        loss_val = flo_loss_val

        # # Compute reconstruction loss
        # rec_loss_val = ((outs["xs_rec"] - batch["x"]) ** 2).mean()
        # rec_loss_val = rec_loss_val * model_config["training"]["loss"]["mse_scale"]
        # alpha = model_config["training"]["loss"]["alpha"]
        # loss = alpha * rec_loss_val + (1 - alpha) * loss_val

        # # Compute weight decay loss
        # weight_loss_val = l2_weight_loss(params["params"])
        # loss_val = loss_val + weight_loss_val

        # # Compute sparsity
        sparsity_val = (outs_1["z"].mean() + outs_2["z"].mean()) / 2

        metrics = {
            "loss/total": loss_val,
            "loss/flo": flo_loss_val,
            # "loss/rec": rec_loss_val,
            # "loss/weights": weight_loss_val,
            "sparsity": sparsity_val,
        }

        others = {
            "mutable_updates": jax.tree.map(
                lambda x, y: (x + y) / 2.0, mutable_updates_1, mutable_updates_2
            ),
        }

        return loss_val, (metrics, others)

    # compute loss
    loss_val, (metrics, others) = loss_fn(state["variables"]["params"])

    return metrics


print("\t Test one train step and one eval step")

(xs_1, xs_2), labels = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])
key_1, key_2 = jax.random.split(key)
batch = {
    "x_1": xs_1,
    "key_1": key_1,
    "x_2": xs_2,
    "key_2": key_2,
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


def activation_to_img(z):
    # compute width and height to get an approximate square
    n_height = np.sqrt(z.shape[-1]).astype(int)
    n_width = z.shape[-1] // n_height + int(not (z.shape[-1] % n_height) == 0)
    # pad z with zero if necessary, so we can then reshape the feature dimension (last)
    # to the given width and height
    pad_width = [(0, 0)] * (len(z.shape) - 1)
    pad_width.append((0, n_height * n_width - z.shape[-1]))
    z = np.pad(z, pad_width, mode="constant", constant_values=0.0)
    # also a dummy final dimension, like it's grayscale
    z = z.reshape((*z.shape[:-1], n_height, n_width, 1))
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

    for epoch in tqdm(range(model_config["training"]["epochs"])):

        epoch_metrics = {k: 0.0 for k in metrics.keys()}
        epoch_metrics_val = {k: 0.0 for k in metrics_val.keys()}
        best_val_epoch = {
            "epoch": None,
            "metrics": {k: None for k in metrics_val.keys()},
        }

        for batch_idx, ((xs_1, xs_2), labels) in tqdm(
            enumerate(dataloader_train), leave=False, total=len(dataloader_train)
        ):

            key, key_1, key_2 = jax.random.split(key, 3)
            batch = {
                "x_1": xs_1,
                "key_1": key_1,
                "x_2": xs_2,
                "key_2": key_2,
            }
            state, metrics, grads = train_step(state, batch)
            LAST_STEP = state["step"].item()

            # Log stats for the train batch
            for metric, value in metrics.items():
                writer.add_scalar(
                    metric + "/train/batch", value.item(), global_step=LAST_STEP
                )
                epoch_metrics[metric] += value

            # test on validation set
            if batch_idx % model_config["validation"]["eval_interval"] == 0:
                (xs_1_val, xs_2_val), labels_val = next(iter(dataloader_val))
                batch_val = {
                    "x_1": xs_1_val,
                    "x_2": xs_2_val,
                }
                metrics_val = eval_step(state, batch_val)
                for metric, value in metrics_val.items():
                    writer.add_scalar(
                        metric + "/val/batch", value.item(), global_step=LAST_STEP
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
                metric + "/train/epoch", value.item(), global_step=epoch_step
            )
        for metric, value in epoch_metrics_val.items():
            writer.add_scalar(
                metric + "/val/epoch", value.item(), global_step=epoch_step
            )

        # Save checkpoint
        if model_config["training"]["checkpoint"]["save"]:
            # save checkpoint
            checkpoint_manager.save(
                step=epoch_step,
                args=orbax.checkpoint.args.StandardSave(state),
            )

        # take images from dataloader_original, perform a forward pass, and save to tensorboard as image
        xs_original, labels_original = next(iter(dataloader_original))
        outs, _ = forward_eval_jitted(state["variables"], xs_original)
        # convert to images
        activation_img = activation_to_img(outs["z"])
        for i in range(activation_img.shape[0]):
            writer.add_image(
                f"activation/img/{i+1}",
                activation_img[i],
                global_step=epoch_step,
                dataformats="HWC",
            )


except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()
