import os
import argparse

from functools import partial

from time import time
from datetime import datetime
from typing import Callable, Any

from pprint import pprint
import toml

import numpy as onp

import infomax_sbdr as sbdr

from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms as tv_transforms

from tqdm import tqdm

import datasets

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
)
base_folder = os.path.normpath(base_folder)

default_model = "tmp"
default_number = "1"
default_dinov2_folder = os.path.join(
    base_folder,
    "resources",
    "models",
    "pretrained",
    "dinov2_downloads",
)


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
    default=0,
)

parser.add_argument(
    "-dv2",
    "--dinov2_folder",
    type=str,
    help="Folder where the DINOv2 model is stored",
    default=default_dinov2_folder,
)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] =  args.cuda_devices

# set torch device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

"""----------------------"""
""" Model Parameters """
"""----------------------"""

# Model parameters
dino_name = "dinov2_vits14"

dino_pretrained_path = os.path.join(
    args.dinov2_folder,
    f"{dino_name}.pth"
)

dino_hubconf_dir = os.path.join(
    args.dinov2_folder,
    "dinov2"
)