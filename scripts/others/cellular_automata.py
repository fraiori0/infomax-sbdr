import jax
import jax.numpy as np

import os
import jax
from jax import vmap, jit, grad
import jax.numpy as np
import numpy as onp
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import infomax_sbdr as sbdr
from tqdm import tqdm
from functools import partial
import cv2

np.set_printoptions(precision=4, suppress=True)
pio.renderers.default = "browser"

SEED = 986

# we implement a 2D topology of size NxN
N = 32
N_STEPS = 1000
P_IN = 0.05

# Parameters for CA rule (survival, birth)
N_S = (1,2,3,4)
N_B = (5,)

# neighborood kernel
K = np.array(
    ((1,1,1,1,1),
    (1,1,1,1,1),
    (1,1,0,1,1),
    (1,1,1,1,1),
    (1,1,1,1,1),)
)

HEIGHT = 480
WIDTH = 480

"""-------------------"""
""" Utils """
"""-------------------"""

def init_state(key, N):
    return {
        "z" : np.zeros((N,N), dtype=np.float32),
        "z_in" : np.zeros((N,N), dtype=np.float32),
    }

@jit
def draw_input(key):
    # draw from a bernoulli
    z_in = jax.random.bernoulli(key, p=P_IN, shape=(N,N))
    return z_in


@jit
def step(state, z_in):
    z_prev = state["z"]
    # first apply input units (OR)
    z = 1 - (1 - z_prev) * (1 - z_in)

    # pad borders like we are on a 2d torus, repeating values 
    # from the opposite side
    k_pad = K.shape[0] // 2
    z_pad = np.pad(z, ((k_pad,k_pad),(k_pad,k_pad)), mode="wrap")

    # Compute convolution with kernel
    z_count = jax.scipy.signal.convolve2d(
        z_pad,
        K,
        mode="valid"
    )

    # # # Activate using cellular automata rule
    # Survival
    z_survive = z_count <= 4
    # Birth
    z_birth = z_count == 5
    # Update
    z = z_survive * z_prev + z_birth
    z = z.astype(np.float32)

    new_state = {
        "z" : z,
        "z_in" : z_in
    }
    out = {
        "z" : z,
        "z_in" : z_in,
        "z_count" : z_count,
    }

    return new_state, out
    


def state_to_img(state):
    
    COLOR_1 = onp.array((20,20,55), dtype=onp.uint8)
    COLOR_0 = onp.array((255,220,220), dtype=onp.uint8)
    
    imgs = {}
    for k in ["z", "z_in"]:
        v = state[k]
        # convert the binary state to an upsampled image
        v = onp.array(v)
        v = v * 255
        v = v.astype(onp.uint8)
        img = COLOR_0 * (1 - v)[..., None] + COLOR_1 * v[..., None]
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        imgs[k] = img

    return imgs


"""-------------------"""
""" Simulate """
"""-------------------"""

key = jax.random.key(SEED)
state = init_state(key, N)


# init history
history = {
    k : [] for k in state.keys()
}

for i in tqdm(range(N_STEPS)):
    
    # update random key
    key, _ = jax.random.split(key)

    # draw input
    z_in = draw_input(key)

    # update state
    state, _ = step(state, z_in)

    # collect state
    for k in state.keys():
        history[k].append(state[k])

# convert to numpy array
for k in history.keys():
    history[k] = np.array(history[k])




