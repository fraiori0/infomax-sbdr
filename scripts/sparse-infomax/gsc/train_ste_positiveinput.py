import os
import argparse

default_model = "ste"
default_number = "1"
default_cuda = "0"

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

# print the shapes nicely
def get_shapes(nested_dict):
    return jax.tree_util.tree_map(lambda x: x.shape, nested_dict)

"""---------------------"""
""" Import model config """
"""---------------------"""
model_folder = os.path.join(
    base_folder,
    "resources",
    "models",
    "gsc",
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
    "gsc",
)

MAX_SAMPLES: int | None = None # 2050          # ← set to e.g. 500 for quick testing
CACHE_DIR:   str | None = "./cache"     # ← set to None to skip disk caching

transform = sbdr.SpecAugmentTransform(
    **model_config["dataset"]["transform"]["specaugment"]["kwargs"],
)

dataset_train = sbdr.GSCDataset(
    root        = data_folder,
    split       = "train",
    precompute  = True,
    augment     = None,
    max_samples = MAX_SAMPLES,
    cache_dir   = CACHE_DIR,
)

dataset_val = sbdr.GSCDataset(
    root        = data_folder,
    split       = "val",
    precompute  = True,
    augment     = None,
    max_samples = MAX_SAMPLES,
    cache_dir   = CACHE_DIR,
)

dataloader_train = sbdr.NumpyLoader(
    dataset_train,
    batch_size=model_config["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
    generator=torch.Generator().manual_seed(model_config["model"]["seed"]),
)

dataloader_val = sbdr.NumpyLoader(
    dataset_val,
    batch_size=model_config["validation"]["dataloader"]["batch_size"],
    shuffle=False,
    drop_last=True,
    generator=torch.Generator().manual_seed(model_config["model"]["seed"]),
)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"Train samples : {len(dataset_train):>7,}   batches: {len(dataloader_train):,}")
print(f"Val   samples : {len(dataset_val):>7,}   batches: {len(dataloader_val):,}")
# print(f"Spectrogram   : ({N_MELS} mels × {N_FRAMES} frames)")
print(f"Cache dir     : {CACHE_DIR}")


print("\nSample shapes")
xs, labels = next(iter(dataloader_train))
print(f"xs: {xs.shape}")
print(f"labels: {labels.shape}")

# # Compute maximum and minimum per each feature across whole dataset 
# xmin = np.inf
# xmax = -np.inf
# for xs, _ in tqdm(dataloader_train, desc="Computing dataset min and max"):
#     batch_min = xs.reshape((-1, xs.shape[-1])).min(axis=0)
#     batch_max = xs.reshape((-1, xs.shape[-1])).max(axis=0)
#     xmin = np.minimum(xmin, batch_min)
#     xmax = np.maximum(xmax, batch_max)

# print(f"Min: {xmin}")
# print(f"Max: {xmax}")
# print(np.maximum(np.abs(xmin), np.abs(xmax)))


"""---------------------"""
""" Init Network """
"""---------------------"""

print("\nInitializing model")

model_class = sbdr.config_rpl_module_dict[model_config["model"]["type"]]

model = model_class(
    **model_config["model"]["kwargs"],
)

model_eval = model_class(
    **model_config["model"]["kwargs"],
)

# # # Initialize parameters
# Take some data
x_seq, labels = next(iter(dataloader_train))
# Generate key
key = jax.random.key(model_config["model"]["seed"])
print(key)
print(key.shape)
# Init recurrent state
s0 = model.init_state_from_input(key, x_seq[..., 0, :])
pprint(get_shapes(s0))

# Init params
variables = model.init(key, s0, x_seq[..., 0, :])

print(f"\tDict of variables: \n\t{variables.keys()}")

pprint(get_shapes(variables))


"""---------------------"""
""" Forward Pass """
"""---------------------"""

print("\nForward pass jitted")


def forward(variables, xs, key):
    # init state
    s0 = model.init_state_from_input(key, xs[..., 0, :])
    # # rescale xs from [-10, 10] to [0, 1]
    # xs = (xs + 10) / 20
    # xs = np.clip(xs, 0.0, 1.0)
    # scan over the sequence
    out = model.apply(
        variables,
        s0,
        xs,
        method=model.scan,
    )
    return out


def forward_eval(variables, xs, key):
    # init state
    s0 = model_eval.init_state_from_input(key, xs[..., 0, :])
    # # rescale xs from [-10, 10] to [0, 1]
    # xs = (xs + 10) / 20
    # xs = np.clip(xs, 0.0, 1.0)
    # scan over the sequence
    out = model_eval.apply(
        variables,
        s0,
        xs,
        method=model_eval.scan,
    )
    return out


forward_jitted = jit(forward)
forward_eval_jitted = jit(forward_eval)

# test the forward pass
xs, labels = next(iter(dataloader_train))
# xs_1 = jax.device_put(xs_1)
# xs_2 = jax.device_put(xs_2)
key = jax.random.key(model_config["model"]["seed"])

# BATCH_NORM - change here
# outs, mutable_updates = forward_jitted(variables, xs_1, key)
# _, _ = forward_eval_jitted(variables, xs_1)
outs = forward_jitted(variables, xs, key)
_ = forward_eval_jitted(variables, xs, key)

print(f"\tInput shape: {xs.shape}")
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

# for k in outs[0].keys():
#     print(f"\n{k}")
#     print(f"\tmin: {outs[0][k].min()}")
#     print(f"\tmax: {outs[0][k].max()}")
#     print(f"\tmean: {outs[0][k].mean()}")
#     print(f"\tstd: {outs[0][k].std()}")


"""---------------------"""
""" Loss Function """
"""---------------------"""

sim_fn = partial(
    sbdr.config_similarity_dict[model_config["training"]["loss"]["sim_fn"]["type"]],
    **model_config["training"]["loss"]["sim_fn"]["kwargs"],
)

def w_infonce(w):
    # Compute an InfoNCE loss with custom critic, given a weight matrix 
    # the weight matrix is assumed to be of shape (in_features, out_features)
    # and InfoNCE is computed to separate weights across units

    eps = 0.01 # model_config["training"]["loss"]["sim_fn"]["kwargs"]["eps"]

    w = w.T # reshape to (out_features, in_features)
    w_avg = w.mean(0)

    p_ii = (w*w).sum(-1) + eps
    p_avg = (w*w_avg).sum(-1) + eps
    w_loss = -np.log(p_ii / p_avg).mean()

    aux = {
        "norm" : np.linalg.norm(w, axis=-1).mean()
    }

    return w_loss, aux


def encoder_infonce(a):
    # given a vector of non-negative values
    # and shape (*batch_dims, time, features)
    # compute a predictive InfoNCE loss

    eps = model_config["training"]["loss"]["sim_fn"]["kwargs"]["eps"]
    pred_horizon = model_config["training"]["pred_horizon"]

    # Compute average activation (for contrasting)
    a_avg = a.reshape((-1, a.shape[-1])).mean(0)
    
    # Compute InfoNCE loss to separate activations across al samples
    p_ii = (a * a).sum(-1) + eps
    p_avg = (a * a_avg).sum(-1) + eps
    per_sample_loss = -np.log(p_ii / p_avg).mean()
    # Compute InfoNCE to separate average activity (in time) across sequences
    a_time = a.mean(-2)
    p_ii_time = (a_time * a_time).sum(-1) + eps
    p_avg_time = (a_time * a_avg).sum(-1) + eps
    time_loss = -np.log(p_ii_time / p_avg_time).mean()

    loss_val = per_sample_loss + 0.5 * time_loss

    aux = {
        "sample" : per_sample_loss,
        "time" : time_loss,
    }

    return loss_val, aux

def memory_infonce(z_query, s_query, s):
    zq_avg = z_query.reshape((-1, z_query.shape[-1])).mean(0)
    sq_avg = s_query.reshape((-1, s_query.shape[-1])).mean(0)
    s_avg = s.reshape((-1, s.shape[-1])).mean(0)

    s_time = s.mean(-2)

    eps = model_config["training"]["loss"]["sim_fn"]["kwargs"]["eps"]

    p_szq = (s * z_query).sum(-1) + eps
    p_szq_avg = (s * zq_avg).sum(-1) + eps
    p_ssq = (s_query * s).sum(-1) + eps 
    p_ssq_avg = (s_query * s_avg).sum(-1) + eps

    z_loss = -np.log(p_szq / p_szq_avg).mean()
    s_loss = -np.log(p_ssq / p_ssq_avg).mean()

    p_ii_time = (s_time * s_time).sum(-1) + eps
    p_avg_time = (s_time * s_avg).sum(-1) + eps
    time_loss = -np.log(p_ii_time / p_avg_time).mean()

    rho = 0.8
    loss_val = rho * z_loss + (1 - rho) * s_loss + 0.2 * time_loss

    aux = {
        "z" : z_loss,
        "s" : s_loss,
        "time" : time_loss
    }

    return loss_val, aux

def timecontrast_infonce(s, z):

    eps = model_config["training"]["loss"]["sim_fn"]["kwargs"]["eps"]

    # Compute global average
    s_avg = s.reshape((-1, s.shape[-1])).mean(0)

    # Compute averages on different time scales
    horizons = [2, 5, 9, 17, 33]
    s_time = []
    for h in horizons:
        w = np.ones((h,)) / h
        s_time.append(sbdr.conv1d(s, w, axis=-2, mode="valid"))
    
    # # append the global average
    # s_time.append(s_avg)

    # Contrast each sammple using positives from 
    # a time scale and negatives from the next time scale
    s_loss = []
    loss_val = 0.0
    for i in range(len(s_time)-1):
        pos_avg = s_time[i]
        neg_avg = s_time[i+1]
        # cut to have same length
        i_start = horizons[i+1] // 2
        i_end = s.shape[-2] - i_start
        s_query = z[..., i_start:i_end, :]
        pos_avg = pos_avg[..., i_start:i_end, :]
        # compute InfoNCE
        p_ii = (s_query * pos_avg).sum(-1) + eps
        p_avg_s = (s_query * neg_avg).sum(-1) + eps
        s_loss_val = -np.log(p_ii / p_avg_s).mean()
        s_loss.append(s_loss_val)
        loss_val += s_loss_val

    # add also individual samples against globla average


    aux = {f"s_t{k:02d}" : s_loss[i] for i, k in enumerate(horizons[:-1])}

    return loss_val, aux

def timepred_infonce(s, a):

    eps = model_config["training"]["loss"]["sim_fn"]["kwargs"]["eps"]

    # Compute global averages
    s_avg = s.reshape((-1, s.shape[-1])).mean(0)
    a_avg = a.reshape((-1, a.shape[-1])).mean(0)

    # positive are s and a
    p_ii = (s[..., :-1, :] * a[..., 1:, :]).sum(-1) + eps
    p_avg_s = (s[..., :-1, :] * a_avg).sum(-1) + eps
    p_avg_a = (a[..., 1:, :] * s_avg).sum(-1) + eps
    s_loss = -np.log(p_ii / p_avg_s).mean()
    a_loss = -np.log(p_ii / p_avg_a).mean()

    loss_val = s_loss + a_loss

    aux = {
        "s" : s_loss,
        "a" : a_loss,
    }

    return loss_val, aux

def timeaverage_infonce(s, z):

    eps = model_config["training"]["loss"]["sim_fn"]["kwargs"]["eps"]

    # Compute global average
    s_avg = s.reshape((-1, s.shape[-1])).mean(0)
    z_avg = z.reshape((-1, z.shape[-1])).mean(0)

    # Compute local neighborood average
    ws = np.linspace(0, 1, num=5)[1:]
    ws = np.concatenate((ws, np.ones((1,)), ws[::-1]))
    ws = ws / ws.sum()
    pos_avg = sbdr.conv1d(z, ws, axis=-2, mode="same")
    neg_avg = 0.5 * (s_avg + z_avg)

    p_ii = (s * pos_avg).sum(-1) + eps
    p_avg_s = (s * neg_avg).sum(-1) + eps
    s_loss = -np.log(p_ii / p_avg_s).mean()

    loss_val = s_loss

    aux = {
        "s" : s_loss,
    }

    return loss_val, aux
    
    
def homeostatic_loss(z):
    p_target = model_config["model"]["p_target"]
    z_avg = z.reshape((-1, z.shape[-1])).mean(0)
    z_avg_sample = z.reshape((-1, z.shape[-1])).mean(-1)
    
    # Note, in the forward we setup z to have a gradient of 1 with respect to the bias
    delta_p = ((p_target - z_avg)**2).mean()

    delta_p_sample = np.clip((z_avg_sample - 2 * p_target)**2, 0, None).mean()

    aux = {
        "delta_p": delta_p,
        "delta_p_sample": delta_p_sample,
    }

    loss_val = delta_p + delta_p_sample

    return delta_p,aux


def classification_loss(logits, labels):
    # Average logits over time
    logits = logits.mean(-2)
    # Compute one-hot labels
    labels_onehot = jax.nn.one_hot(labels, logits.shape[-1])
    # Compute categorical crossentropy loss
    loss = -(labels_onehot * jax.nn.log_softmax(logits)).sum(axis=-1)
    # compute accuracy
    aux = {
        "acc": (labels[:, None] == logits.argmax(-1)).mean(),
    }
    return loss.mean(), aux


def flo_loss(
    outs_layer_1,
    outs_layer_2,
):

    eps = model_config["training"]["loss"]["sim_fn"]["kwargs"]["eps"]

    a1 = outs_layer_1["a"]
    z1 = outs_layer_1["z"]
    a2 = outs_layer_2["a"]
    z2 = outs_layer_2["z"]

    za1 = z1*a1
    za2 = z2*a2
    za1_avg = za1.reshape((-1, za1.shape[-1])).mean(0)
    za2_avg = za2.reshape((-1, za2.shape[-1])).mean(0)

    # steps ahead to make the prediction
    pred_horizon = model_config["training"]["pred_horizon"]

    # # # Prediction infoNCE
    p_ii_12 = (za1[..., :-pred_horizon, :] * jax.lax.stop_gradient(za2[..., pred_horizon:, :])).sum(-1) + eps
    p_ii_21 = (za2[..., :-pred_horizon, :] * jax.lax.stop_gradient(za1[..., pred_horizon:, :])).sum(-1) + eps
    p_avg_12 = (za1[..., :-pred_horizon, :] * za2_avg).sum(-1) + eps
    p_avg_21 = (za2[..., :-pred_horizon, :] * za1_avg).sum(-1) + eps

    flo_val_12 = -np.log(p_ii_12 / p_avg_12).mean()
    flo_val_21 = -np.log(p_ii_21 / p_avg_21).mean()
    flo_val = 0.5 * (flo_val_12 + flo_val_21)

    aux = {
        "flo_12": flo_val_12,
        "flo_21": flo_val_21,
    }

    return flo_val, aux

flo_loss_jitted = jit(flo_loss)

# test loss function
xs, labels = next(iter(dataloader_train))
key = jax.random.key(model_config["model"]["seed"])
outs = forward(variables, xs, key)
w_loss_val, aux = w_infonce(variables["params"]["layers_0"]["w"])
a_loss_val, aux = encoder_infonce(outs[0]["a"])
# m_loss_val, aux = memory_infonce(outs[0]["z_query"], outs[0]["s_query"], outs[0]["s"])
t_loss_val, aux = timeaverage_infonce(outs[0]["s"], outs[0]["z"])

print(f"\tLosses: w: {w_loss_val}, a: {a_loss_val}, t: {t_loss_val}")

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


# Overwrite memory/gate params with the initial ones
# state["variables"]["params"]["layers_0"]["mem"]["kernel"] = variables["params"]["layers_0"]["mem"]["kernel"]
# state["variables"]["params"]["layers_0"]["mem"]["bias"] = variables["params"]["layers_0"]["mem"]["bias"]
# state["variables"]["params"]["layers_0"]["gate"]["kernel"] = variables["params"]["layers_0"]["gate"]["kernel"]
# state["variables"]["params"]["layers_0"]["gate"]["bias"] = variables["params"]["layers_0"]["gate"]["bias"]


# overwrite "variables"
variables = state["variables"]


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

    for l_idx, l_out in enumerate(outs):
        for k in ["a", "z", "s", "zs"]:
            
            # per-unit average
            unit_avg = np.mean(l_out[k].reshape((-1, l_out[k].shape[-1])), axis=0)
            qs_unit_val = np.quantile(unit_avg, QS)

            # per-sample average
            sample_avg = np.mean(l_out[k].reshape((-1, l_out[k].shape[-1])), axis=-1)
            qs_sample_val = np.quantile(sample_avg, QS)


            metrics.update(
                {f"unit_{l_idx}_{k}/{kk}": v for kk, v in zip(Q_KEYS, qs_unit_val)}
            )
            metrics.update(
                {f"sample_{l_idx}_{k}/{kk}": v for kk, v in zip(Q_KEYS, qs_sample_val)}
            )

    return metrics



"""---------------------"""
""" Training and Evaluation Steps """
"""---------------------"""

print("\nTraining and Evaluation Steps")


@jit
def train_step(state, batch):

    # index of the layer that we train
    l_index = model_config["training"]["layer_idx"]
    
    def loss_fn(params):
        # Apply the model
        outs = forward(
            {
                "params": params,
            },
            batch["x_1"],
            batch["key_1"],
        )

        # select only outputs from the layer that is being trained
        out_l = outs[l_index]
        l_key = "layers_" + str(l_index)

        # Compute loss for the weights
        # w_loss_val, w_aux = w_infonce(params[l_key]["w"])

        # Compute loss on the forward activations
        # add some positive normal noiseto avoid flat gradient with zero activations
        # a_noise = np.clip(0.05 * jax.random.normal(batch["key_1"], shape=out_l["a"].shape), 0, None)
        a_loss_val, a_aux = encoder_infonce(out_l["a"])

        # # Compute loss on memory state
        # m_loss_val, m_aux = memory_infonce(out_l["z_query"], out_l["s_query"], out_l["s"])

        # Compute loss on time-averaged memory state
        # t_loss_val, t_aux = timeaverage_infonce(out_l["s"], out_l["z"])
        t_loss_val, t_aux = timecontrast_infonce(out_l["e_s"], out_l["e_z"])
        # t_loss_val, t_aux = timepred_infonce(out_l["s"], out_l["a"])
        

        # Compute classification loss
        c_loss_val, c_aux = classification_loss(out_l["logits"], batch["labels"])

        # Compute homeostatic loss
        # h_loss_val, h_aux = homeostatic_loss(out_l["z"])

        # Compute cumulative loss
        loss_val = (
            # w_loss_val * 0.1
            + a_loss_val
            + t_loss_val
            # + h_loss_val * 10
            + c_loss_val
        )
        
        # Store losses
        aux = {
            # "w": w_loss_val,
            "a": a_loss_val,
            # "h": h_loss_val,
            "t": t_loss_val,
            "c": c_loss_val,
        }

        # compute useful metrics for logging
        metrics = compute_metrics(outs)
        # add flo and total loss
        for k, v in aux.items():
            metrics.update({f"aux_losses_{l_index}/{k}": v})

        for k, v in t_aux.items():
            metrics.update({f"aux_losses_{l_index}/t_{k}": v})
        for k, v in a_aux.items():
            metrics.update({f"aux_losses_{l_index}/a_{k}": v})
        for k, v in c_aux.items():
            metrics.update({f"aux_losses_{l_index}/class_{k}": v})
            
        metrics.update({f"loss/{l_index}": loss_val})

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
    new_params = optax.apply_updates(
        state["variables"]["params"], updates
    )
    # Update optimizer state
    state["opt_state"] = opt_state

    # # BATCH_NORM - change here
    # # update batch stats
    # state["variables"]["batch_stats"] = others["mutable_updates"]["batch_stats"]

    # Clip encoder weights
    l_key = "layers_" + str(l_index)
    # w_max = 1.0 # /np.sqrt(new_params[l_key]["w"].shape[-1]) # 1/sqrt(fan_out)
    # new_params[l_key]["w"] = np.clip(new_params[l_key]["w"], 0.0, w_max)

    # Set new params
    state["variables"]["params"] = new_params

    # update step
    state["step"] += 1

    return state, metrics, grads


@jit
def eval_step(state, batch):

    l_index = model_config["training"]["layer_idx"]
    
    def loss_fn(params):
        # Apply the model
        outs = forward_eval(
            {
                "params": params,
            },
            batch["x_1"],
            batch["key_1"],
        )

        # select only outputs from the layer that is being trained
        out_l = outs[l_index]
        l_key = "layers_" + str(l_index)

        # Compute loss for the weights
        # w_loss_val, w_aux = w_infonce(params[l_key]["w"])

        # Compute loss on the forward activations
        # add some positive normal noiseto avoid flat gradient with zero activations
        # a_noise = np.clip(0.05 * jax.random.normal(batch["key_1"], shape=out_l["a"].shape), 0, None)
        a_loss_val, a_aux = encoder_infonce(out_l["a"])

        # # Compute loss on memory state
        # m_loss_val, m_aux = memory_infonce(out_l["z_query"], out_l["s_query"], out_l["s"])

        # Compute loss on time-averaged memory state
        # t_loss_val, t_aux = timeaverage_infonce(out_l["s"], out_l["z"])
        t_loss_val, t_aux = timecontrast_infonce(out_l["e_s"], out_l["e_z"])
        # t_loss_val, t_aux = timepred_infonce(out_l["s"], out_l["a"])
        

        # Compute classification loss
        c_loss_val, c_aux = classification_loss(out_l["logits"], batch["labels"])

        # # Compute homeostatic loss
        # h_loss_val, h_aux = homeostatic_loss(out_l["z"])

        # Compute cumulative loss
        loss_val = (
            # w_loss_val * 0.1
            # + a_loss_val
            + c_loss_val
            + t_loss_val
            # + h_loss_val * 10
        )
        
        # Store losses
        aux = {
            # "w": w_loss_val,
            "a": a_loss_val,
            # "h": h_loss_val,
            "t": t_loss_val,
            "c": c_loss_val,
        }

        # compute useful metrics for logging
        metrics = compute_metrics(outs)
        # add flo and total loss
        for k, v in aux.items():
            metrics.update({f"aux_losses_{l_index}/{k}": v})

        for k, v in t_aux.items():
            metrics.update({f"aux_losses_{l_index}/t_{k}": v})
        for k, v in a_aux.items():
            metrics.update({f"aux_losses_{l_index}/a_{k}": v})
        for k, v in c_aux.items():
            metrics.update({f"aux_losses_{l_index}/class_{k}": v})
            
        metrics.update({f"loss/{l_index}": loss_val})

        others = {
            # BATCH_NORM - change here
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
    "x_1": xs,
    "key_1": key_1,
    "x_2": xs,
    "key_2": key_2,
    "labels": labels,
}

state, metrics, grads = train_step(state, batch)
metrics_val = eval_step(state, batch)

print(f"\tmetrics: {metrics}")

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
@jit
def compute_activation_imgs(outs, params):
    """Compute metrics from model outputs."""
    img_dict = {}

    N_MAX = 8

    for l_idx, l_out in enumerate(outs):
        # # binary activations
        for k_z, k_a, k_s, k_zs in zip(["z",], ["a",], ["s",], ["zs",]):
            
            # select only a few outputs (e.g., 8) from the batch size
            _z = l_out[k_z][:N_MAX]
            _a = l_out[k_a][:N_MAX]
            _s = l_out[k_s][:N_MAX]
            _zs = l_out[k_zs][:N_MAX]

            # z is already in 0-1, a needs to be 
            # put to 0-1
            # _a = (_a - _a.min()) / (_a.max() - _a.min())

            # add last channel (grayscale)
            _z = _z.reshape((*_z.shape, 1))
            _a = _a.reshape((*_a.shape, 1))
            _s = _s.reshape((*_s.shape, 1))
            _zs = _zs.reshape((*_zs.shape, 1))

            img_dict.update(
                {f"activation_{l_idx}_{k_z}/{img_i}": img for img_i, img in enumerate(_z)}
            )
            img_dict.update(
                {f"activation_{l_idx}_{k_a}/{img_i}": img for img_i, img in enumerate(_a)}
            )
            img_dict.update(
                {f"activation_{l_idx}_{k_s}/{img_i}": img for img_i, img in enumerate(_s)}
            )
            img_dict.update(
                {f"activation_{l_idx}_{k_zs}/{img_i}": img for img_i, img in enumerate(_zs)}
            )

        # Add encoder weights as images
        w = params["layers_" + str(l_idx)]["w"]
        w = w.T # reshape to (out_features, in_features)
        # reshape in_features to be a square if possible, pad with 0 if necessary
        w_img = w.reshape((w.shape[0], 8, 5, 1))
        w_img = (w_img - w_img.min()) / (w_img.max() - w_img.min())
        for unit_i in range(min(w.shape[0], N_MAX)):
            img_dict[f"weights_{l_idx}/unit_{unit_i}"] = w_img[unit_i]

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

    # take images from dataloader_original, perform a forward pass, and save to tensorboard as image
    xs_original, labels_original = next(iter(dataloader_val))
    # BATCH_NORM - change here
    # outs, _ = forward_eval_jitted(state["variables"], xs_original)
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

        for batch_idx, (xs, labels) in tqdm(
            enumerate(dataloader_train), leave=False, total=len(dataloader_train)
        ):

            key, subkey = jax.random.split(key, 2)
            batch = {
                "x_1": xs,
                "key_1": key,
                "x_2": xs,
                "key_2": subkey,
                "labels": labels,
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
                xs, labels_val = next(iter(dataloader_val))
                key, subkey = jax.random.split(key, 2)
                batch_val = {
                    "x_1": xs,
                    "key_1": key,
                    "x_2": xs,
                    "key_2": subkey,
                    "labels": labels_val,
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
        xs_original, labels_original = next(iter(dataloader_val))
        # BATCH_NORM - change here
        # outs, _ = forward_eval_jitted(state["variables"], xs_original)
        key, _ = jax.random.split(key, 2)
        outs = forward_eval_jitted(state["variables"], xs_original, key)
        # convert to images
        img_dict = compute_activation_imgs(outs, state["variables"]["params"])
        for k, v in img_dict.items():
            v = onp.array(v) # important, tensorboard wants lists or numpy arrays, not jax.numpt arrays
            writer.add_image(k, v, global_step=epoch_step, dataformats="HWC")

except KeyboardInterrupt:
    print("\nTraining interrupted")

writer.close()
