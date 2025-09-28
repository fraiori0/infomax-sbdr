import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "3"

import jax
import jax.numpy as np
import numpy as onp
from tbparse import SummaryReader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.set_printoptions(precision=4, suppress=True)

default_model = "vgg_gavg_sigmoid_logand" #"vgg_sigmoid_and"  # "vgg_sbdr_5softmax/1"  #
default_number = "1"

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    os.pardir,
)
base_folder = os.path.normpath(base_folder)


"""---------------------"""
""" Import data """
"""---------------------"""


log_folder = os.path.join(
    base_folder,
    "resources",
    "models",
    "cifar10",
    default_model,
    default_number,
    "logs",
)


reader = SummaryReader(log_folder)
df = reader.scalars
print(df)