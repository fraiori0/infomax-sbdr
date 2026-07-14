import os
import argparse

default_model = "dense_sparse_dictionary"
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
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
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

"""---------------------"""
""" Import model config """
"""---------------------"""
model_folder = os.path.join(
    base_folder,
    "resources",
    "models",
    "fashionmnist",
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
        # normalize
        tv_transforms.Normalize(
            mean=model_config["dataset"]["transform"]["normalization"]["mean"],
            std=model_config["dataset"]["transform"]["normalization"]["std"],
        ),
        # change from  (C, H, W) to (H, W, C)
        tv_transforms.Lambda(lambda x: x.movedim(-3, -1)),
    ]
)

dataset = sbdr.FashionMNISTDatasetContrastive(
    folder_path=data_folder,
    kind="train",
    transform=transform,
    flatten=True,
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


(xs_1, xs_2), labels = next(iter(dataloader_train))
# print original device
print(f"\tOriginal device: {xs_1.device}")
print(f"\tOriginal device: {xs_2.device}")
print()
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
dataset_original = sbdr.FashionMNISTDataset(
    folder_path=data_folder,
    kind="train",
    transform=transform_original,
    flatten=True,
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


def untransform(x):
    return (x * np.array([0.3530])) + np.array([0.2860])


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

model_class = sbdr.config_module_dict[model_config["model"]["type"]]

model = model_class(
    **model_config["model"]["kwargs"],
    activation_fn=sbdr.config_activation_dict[model_config["model"]["activation"]],
    # out_activation_fn=sbdr.config_activation_dict[
    #     model_config["model"]["out_activation"]
    # ],
    training=True,
)

model_eval = model_class(
    **model_config["model"]["kwargs"],
    activation_fn=sbdr.config_activation_dict[model_config["model"]["activation"]],
    # out_activation_fn=sbdr.config_activation_dict[
    #     model_config["model"]["out_activation"]
    # ],
    training=False,
)

# # # Initialize parameters
# Take some data
(xs_1, xs_2), labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
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
        # # BATCH_NORM
        # # batch stats should be updated
        # mutable=["batch_stats"],
    )


def forward_eval(variables, xs, key):
    return model_eval.apply(
        variables,
        xs,
        # # BATCH_NORM change here
        # mutable=["batch_stats"],
    )


forward_jitted = jit(forward)
forward_eval_jitted = jit(forward_eval)

# test the forward pass
(xs_1, xs_2), labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
key = jax.random.key(model_config["model"]["seed"])

# BATCH_NORM - change here
# outs, mutable_updates = forward_jitted(variables, xs_1, key)
# _, _ = forward_eval_jitted(variables, xs_1)
outs = forward_jitted(variables, xs_1, key)
_ = forward_eval_jitted(variables, xs_1, key)

print(f"\tInput shape: {xs_1.shape}")
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
""" Loss Functions """
"""---------------------"""

sim_fn = partial(
    sbdr.config_similarity_dict[model_config["training"]["loss"]["sim_fn"]["type"]],
    **model_config["training"]["loss"]["sim_fn"]["kwargs"],
)


def mse(pred, target):
    loss_val = ((pred - target) ** 2).sum(-1).mean()
    aux = {"mse": loss_val}
    return loss_val, aux

def gaussian_likelihood(pred, target, logvar=1.0):
    se = ((pred - target) ** 2)
    loss_val = logvar + se/(2 * np.exp(logvar))
    loss_val = 0.5 * loss_val.sum()
    aux = {
        "gaussian_likelihood": loss_val,
    }
    return loss_val, aux

def encoder_infonce(a, eps=0.01):

    # Compute average activation (for contrasting)
    a_avg = a.reshape((-1, a.shape[-1])).mean(0)
    
    # Compute InfoNCE loss to separate activations across al samples
    p_ii = (a * a).sum(-1) + eps
    p_avg = (a * a_avg).sum(-1) + eps
    loss_val = -np.log(p_ii / p_avg).mean()

    aux = {
        "infonce" : loss_val,
    }

    return loss_val, aux

def cross_infonce(y_1, y_2, eps=0.01):
    # compute InfoNCE between two encodings
    y_1_avg = y_1.reshape((-1, y_1.shape[-1])).mean(0)
    y_2_avg = y_2.reshape((-1, y_2.shape[-1])).mean(0)

    p_ii = (y_1 * y_2).sum(-1) + eps
    p_avg_1 = (y_1 * y_1_avg).sum(-1) + eps
    p_avg_2 = (y_2 * y_2_avg).sum(-1) + eps
    loss_val = -(np.log(p_ii / p_avg_1) + np.log(p_ii / p_avg_2)).mean()

    aux = {
        "infonce" : loss_val,
    }

    return loss_val, aux

def infonce(y, y_target, eps=0.01):
    # compute InfoNCE between two encodings
    y_avg = y.reshape((-1, y.shape[-1])).mean(0)

    p_ii = (y * y_target).sum(-1) + eps
    p_avg = (y * y_avg).sum(-1) + eps
    loss_val = -np.log(p_ii / p_avg).mean()

    aux = {
        "infonce" : loss_val,
    }

    return loss_val, aux

def classification_loss(logits, labels):
    # Compute one-hot labels
    # labels_onehot = jax.nn.one_hot(labels, logits.shape[-1])
    labels_onehot = labels # if already one-hot
    # Compute categorical crossentropy loss
    loss = -(labels_onehot * jax.nn.log_softmax(logits)).sum(axis=-1)
    # # compute accuracy
    # acc = (labels[:, None] == logits.argmax(-1)).mean()
    acc = (labels.argmax(-1) == logits.argmax(-1)).mean()
    aux = {
        "acc": acc,
    }
    return loss.mean(), aux

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

Q_KEYS = ["0.05", "0.25", "0.5", "0.75", "0.95"]
QS = np.array((0.05, 0.25, 0.5, 0.75, 0.95))

@jit
def compute_metrics(outs):
    """Compute metrics from model outputs."""
    metrics = {}

    for k in ["z"]:
        
        # per-unit average
        unit_avg = np.mean(outs[k].reshape((-1, outs[k].shape[-1])), axis=0)
        qs_unit_val = np.quantile(unit_avg, QS)

        # per-sample average
        sample_avg = np.mean(outs[k].reshape((-1, outs[k].shape[-1])), axis=-1)
        qs_sample_val = np.quantile(sample_avg, QS)


        metrics.update(
            {f"unit_{k}/{kk}": v for kk, v in zip(Q_KEYS, qs_unit_val)}
        )
        metrics.update(
            {f"sample_{k}/{kk}": v for kk, v in zip(Q_KEYS, qs_sample_val)}
        )

    return metrics


"""---------------------"""
""" Training and Evaluation Steps """
"""---------------------"""

print("\nTraining and Evaluation Steps")


def loss_fn_gen(state, batch, forward_fn):
    def loss_fn(params):
        # Apply the model
        # BATCH_NORM - change here
        # outs_1, mutable_updates_1 = forward(
        outs_1 = forward_fn(
            {
                "params": params,
                # # BATCH_NORM - change here
                # "batch_stats": state["variables"]["batch_stats"],
            },
            batch["x_1"],
            batch["key_1"],
        )
        # Apply the model to the augmented images
        # BATCH_NORM - change here
        # outs_2, mutable_updates_2 = forward(
        outs_2 = forward_fn(
            {
                "params": params,
                # # BATCH_NORM - change here
                # "batch_stats": state["variables"]["batch_stats"],
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
        
        # Compute loss
        loss_val = 0.0
        # Take values
        z1 = outs_1["z"]
        z2 = outs_2["z"]
        x1_hat = outs_1["x_hat"]
        x2_hat = outs_2["x_hat"]
        x1_target = outs_1["x_original"]
        x2_target = outs_2["x_original"]
        logits1 = outs_1["logits"]
        logits2 = outs_2["logits"]
        
        # Compute InfoNCE loss between the two embeddings
        z_loss_val, z_aux = cross_infonce(z1, z2)
        # # Reconstruction loss
        x1hat_loss_val, x1hat_aux = mse(x1_hat, x1_target)
        x2hat_loss_val, x2hat_aux = mse(x2_hat, x2_target)
        # Classification loss
        class1_loss_val, class1_aux = classification_loss(logits1, batch["label"])
        class2_loss_val, class2_aux = classification_loss(logits2, batch["label"])

        # Compute total loss
        lam = model_config['training']['loss']['lambda']
        loss_val = loss_val + (
            +  lam * 0.5 * (x1hat_loss_val + x2hat_loss_val)
            + (1 - lam) * z_loss_val
            + (class1_loss_val + class2_loss_val) # if there is stop gradient no gradient will flow to z
        )

        # COmpute metrics (only on one output, we need to be careful but they should be similar, it's the same network and same weights)
        metrics = compute_metrics(outs_1)

        metrics.update({f"loss": loss_val})
        for k, v in z_aux.items():
            metrics.update({f"aux_losses/z_{k}": v})
        for k, v in x1hat_aux.items():
            metrics.update({f"aux_losses/x1_{k}": v})
        for k, v in x2hat_aux.items():
            metrics.update({f"aux_losses/x2_{k}": v})
        for k, v in class1_aux.items():
            metrics.update({f"aux_losses/class1_{k}": v})
        for k, v in class2_aux.items():
            metrics.update({f"aux_losses/class2_{k}": v})

        others = {
            # BATCH_NORM - change here
            # "mutable_updates": jax.tree.map(
            #     lambda x, y: (x + y) / 2.0, mutable_updates_1, mutable_updates_2
            # ),
        }

        return loss_val, (metrics, others)

    return loss_fn

@jit
def train_step(state, batch):
    loss_fn = loss_fn_gen(state, batch, forward_fn = forward)

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
    loss_fn = loss_fn_gen(state, batch, forward_fn = forward_eval)

    # compute loss
    loss_val, (metrics, others) = loss_fn(state["variables"]["params"])

    return metrics


print("\t Test one train step and one eval step")

(xs_1, xs_2), labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
key = jax.random.key(model_config["model"]["seed"])
key_1, key_2 = jax.random.split(key)
batch = {
    "x_1": xs_1,
    "key_1": key_1,
    "x_2": xs_2,
    "key_2": key_2,
    "label": labels,
}

state, metrics, grads = train_step(state, batch)
metrics_val = eval_step(state, batch)

print(f"\tTrain metrics:")
pprint(metrics)

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

def activation_to_img(z, normalize=False):
    # # compute width and height to get an approximate square
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
    n_height = 28
    # add a dummy final channel dimension, like it's grayscale
    z = z.reshape((*z.shape[:-1], n_height, -1, 1))
    return z # onp.array(z)

# @jit
def compute_activation_imgs(outs, params):
    """Compute metrics from model outputs."""
    img_dict = {}

    N_MAX = 8

    # # binary activations
    for k in ["z", "x_hat", "x_original"]:
        # select only a few outputs (e.g., 8) from the batch size
        _v = outs[k][:N_MAX]
        # untransform if it is x_hat
        if k in ["x_hat", "x_original"]:
            _v = untransform(_v)
        # convert to square image
        _v = activation_to_img(_v, normalize=False)
        if k in ["z"]:
            _v = _v / model_config["model"]["kwargs"]["n_steps"]

        # add to dictionary for tensorboard
        img_dict.update(
            {f"activation_{k}/{img_i}": img for img_i, img in enumerate(_v)}
        )

    # # Add also sequence of iterative representations, to see how the model behaves
    # _v = outs["z_seq"][:, :N_MAX]
    # _v = np.swapaxes(_v, 0, 1)  # swap batch and sequence dimensions
    # # copy the seq dimension so the image is easier to visualize, otherwise
    # # one step in the sequence is one pixel
    # _v = np.repeat(_v, 4, axis=-1)
    # # add last channel (grayscale)
    # _v = _v.reshape((*_v.shape, 1))
    # img_dict.update(
    #     {f"activation_z_seq/{img_i}": img for img_i, img in enumerate(_v)}
    # )

    return img_dict


"""------------------"""
""" Training """
"""------------------"""

key = jax.random.key(model_config["model"]["seed"])

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
    xs_original, labels_original = next(iter(dataloader_original))
    # BATCH_NORM - change here
    # outs, _ = forward_eval_jitted(state["variables"], xs_original, ke)
    outs = forward_eval_jitted(state["variables"], xs_original, key)
    # convert to images
    img_dict = compute_activation_imgs(outs, state["variables"]["params"])
    for k, v in img_dict.items():
        v = onp.array(v) # important, tensorboard wants lists or numpy arrays, not jax.numpt arrays
        writer.add_image(k, v, global_step=0, dataformats="HWC")

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
                "x_1": xs_1, # jax.device_put(xs_1),
                "key_1": key_1,
                "x_2": xs_2, # jax.device_put(xs_2),
                "key_2": key_2,
                "label": labels, # jax.device_put(labels),
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
                (xs_1_val, xs_2_val), labels_val = next(iter(dataloader_val))
                key, key_1_val, key_2_val = jax.random.split(key, 3)
                batch_val = {
                    "x_1": xs_1_val,
                    "x_2": xs_2_val,
                    "key_1": key_1_val,
                    "key_2": key_2_val,
                    "label": labels_val,
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
        xs_original, labels_original = next(iter(dataloader_original))
        # BATCH_NORM - change here
        # outs, _ = forward_eval_jitted(state["variables"], xs_original)
        outs = forward_eval_jitted(state["variables"], xs_original, key)
        # convert to images
        img_dict = compute_activation_imgs(outs, state["variables"]["params"])
        for k, v in img_dict.items():
            v = onp.array(v) # important, tensorboard wants lists or numpy arrays, not jax.numpt arrays
            writer.add_image(k, v, global_step=epoch_step, dataformats="HWC")


except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()
