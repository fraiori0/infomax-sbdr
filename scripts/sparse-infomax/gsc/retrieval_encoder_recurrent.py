import os
import argparse

default_model = "rec"
default_number = "1"
default_cuda = "1"
CHKP_STEP = 148

data_prefix = "clean"

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
import json

import numpy as onp

import infomax_sbdr as sbdr

from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms as tv_transforms

import cv2
from tqdm import tqdm

from sbdr_retrieval import Database, Engine, ModelConfig, ScalableConfig

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
""" Import encoded (by the model) data """
"""---------------------"""

data_folder = os.path.join(model_folder, "activations")

# Import data from JSON of sparse nested list index of activations
with open(os.path.join(data_folder, f"{data_prefix}_sparse_train.json"), "r") as f:
    train_e_sparse = json.load(f)

with open(os.path.join(data_folder, f"{data_prefix}_sparse_val.json"), "r") as f:
    val_e_sparse = json.load(f)

print("\nLoaded activations from:\n\t", data_folder)

"""---------------------"""
""" Initialize Database for Retrieval """
"""---------------------"""

db_m_config = ModelConfig(
    mode="votes",                  # "votes" or "infonce_global"
    advance_mean=1.0,              # μ: expected positions advanced per frame
    advance_spread=2.0,            # σ: warp tolerance (also sets kernel length)
    stay_prob=0.4,                 # ρ: mass kept at current position (slow/holds)
    emission_floor=0.05,           # ε: per-position emission floor (noise robustness)
    leak_uniform=0.2,              # η: uniform mixing (re-acquisition/recovery)
    infonce_c=0.01,                # c: InfoNCE additive constant (infonce_global only)
    kernel_mode="spec_exact"       # "reference_compat" or "spec_exact"
)

db_s_config = ScalableConfig(
    n_candidates=64,      # C: number of full-state candidates tracked
    prefilter_warmup=15,  # frames of cheap scoring before first prune
    prefilter_every=1,    # refresh the candidate set every N frames
    keep_floor=0,         # (inert; kept for reference compatibility)
)

# # Initialize Database
KEY_ENCODINGS = "zf"
bin_path = os.path.join(model_folder, "activations", "db.bin")
db = Database.build(
    val_e_sparse[KEY_ENCODINGS],
    labels=val_e_sparse["label"],
    d=model_config["model"]["kwargs"]["features"],
)
# save activations
# db.save(bin_path)
# # Load from save, instead
# db = Database.load(bin_path)

# free memory
# del train_e_sparse

# Initialize scalable engine
eng = db.scalable_engine(
    db_m_config,
    db_s_config,
)

# test with some randomsequences from val set (just to see if it works)
key = jax.random.key(0)

for _ in range(20):
    
    # gen random integer index for the sequence
    key, empty = jax.random.split(key)
    i_test = jax.random.randint(key, (1,), 0, len(val_e_sparse["z"])).item()
    print(f"Testing sequence {i_test}")

    # reset engine
    eng.reset()

    seq = val_e_sparse[KEY_ENCODINGS][i_test]
    lab = val_e_sparse["label"][i_test]

    for t in range(len(seq)):
        # get frame
        frame = seq[t]
        frame = onp.array(frame, dtype=onp.uint32)
        # run
        eng.step(frame)
        # score and print to check
        idx, score = eng.best()
        label_pred = eng.predict_label(k=15, tau=0.1)
        # # print so it updates the same line, canceling what was previously written
        # print(f"\r\t{t} | ({idx} : {score:.3f}) - {label_pred}", end="")
        # just print instead
        # if t % 32 == 0:
        #     print(f"\t{t} | ({idx} : {score:.3f}) - {label_pred} true: {lab}")


    print(f"\t{t} | ({idx} : {score:.3f}) - {label_pred} true: {lab}")


