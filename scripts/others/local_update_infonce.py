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

np.set_printoptions(precision=4, suppress=True)
pio.renderers.default = "browser"

SEED = 986
N_FEATURES = 64
N_SAMPLES = 300

EPS = 1e-5
P_DES = 0.05
LR = 2
STEPS = 10000
ALPHA = 0.0

GAMMA = 0.9
# GAMMA = jax.random.uniform(jax.random.key(SEED), shape=(N_FEATURES,), minval=0.3, maxval=0.9)


key = jax.random.PRNGKey(SEED)

@partial(jit, static_argnames=("gamma",))
def eligibility_trace(z, gamma=0.9):
    # compute the eligibility trace of z along axis 0
    def f_scan(carry, input):
        e = carry
        z_t = input
        e = gamma * e * (1 - z_t) + z_t
        return e, e
    e0 = np.zeros(z.shape[-1])
    _, e_trace = jax.lax.scan(f_scan, e0, z)
    return e_trace

# @jit
@partial(jit, static_argnames=("gamma"))
def discounted_trace(z, d0, gamma=0.9):
    # compute the eligibility trace of z along axis 0
    def f_scan(carry, input):
        e = carry
        z_t = input
        e = gamma * e + z_t
        return e, e
    _, d_trace = jax.lax.scan(f_scan, d0, z)
    return d_trace

def encode(z):
    return jax.nn.sigmoid(z)

def expsim(z1, z2, eps=1e-8):
    return (z1*z2).sum(-1)

def infonce(z, z_avg, eps=1e-8):
    return np.log(expsim(z, z_avg) + eps) - np.log(expsim(z, z) + eps)

def crossentropy(z, z_avg, eps=1e-8):
    # alpha = ALPHA
    # eps if z_avg < P_DES, 1-eps if z_avg > P_DES
    alpha = np.where(z_avg < P_DES, ALPHA, 1 - ALPHA)

    h = -(alpha * z_avg * np.log(z + eps) + (1 - alpha) * (1 - z_avg) * np.log(1 - z + eps)).sum(-1)
    
    return -h

def crosslinear(z, z_avg, eps=1e-8):
    # alpha = ALPHA
    alpha = np.where(z_avg < P_DES, ALPHA, 1 - ALPHA)

    h = (1-alpha) * z*(z+0.5) + alpha * (z-1)*(z-1.5)
    
    return -h

def infonce(z, z_avg, eps=1e-8):
    pii = (z*z).sum(-1)
    pij = (z[..., None, :] * z[..., :, None]).sum(-1)
    mi = sbdr.infonce(pii, pij, eps=eps)
    loss_val = -mi
    return loss_val

@jit
def update_samples(z, lr):
    # z of shape (*batch_dims, features)
    # update each z according to it's gradient
    def loss(z):
        z = encode(z)
        z_avg = z.mean(axis=0)
        z_avg = jax.lax.stop_gradient(z_avg)
        # compute eligibility trace
        z_trace = eligibility_trace(z, gamma=GAMMA)
        z_trace_avg = z_trace.mean(axis=0)
        # d0 = z_avg / (1 - GAMMA)
        # z_trace = discounted_trace(z, d0=d0, gamma=GAMMA)*(1 - GAMMA)

        # loss_val = crossentropy(z, z_avg, eps=EPS).sum()
        # loss_val = crosslinear(z, z_avg, eps=EPS).sum()
        # loss_val = crossentropy(z_trace, z_trace_avg, eps=EPS).sum()
        cl = crosslinear(z_trace, z_avg, eps=EPS).sum()
        mi = infonce(z_trace, z_avg, eps=EPS).sum()
        loss_val = mi

        """
        Notes:
            1. [SUCC] crosslinear with eligibility trace and z_avg (i.e., avg on single steps, not on eligibility trace)
                seems to work well to produce binary activity.
                However, does it actually maximize some sort of MI or separation?
            2. [SUCC] also using infonce on eligibility trce brings sparse binary activity;
                however, it happens very slowly and it needs a lot of epochs/steps.
        """

        aux = {
            "z": z,
            "z_avg": z_avg,
            "mi": mi,
        }
        return loss_val, aux

    z_shape = z.shape
    z = z.reshape((-1, z.shape[-1]))

    (loss_val, aux), dz = jax.value_and_grad(loss, has_aux=True)(z)

    z_new = z - lr * dz
    z_new = z_new.reshape(z_shape)

    aux["loss"] = loss_val.mean()
    aux["dz"] = dz.reshape(z_shape)
    aux["z"] = aux["z"].reshape(z_shape)

    return z_new, aux

"""----------------------"""
"""Draw samples"""
"""----------------------"""

print("\nDrawing samples")

key, _ = jax.random.split(key)
z0 = jax.random.uniform(key, shape=(N_SAMPLES, N_FEATURES), minval=-2.0, maxval=2.0)

z_new, aux = update_samples(z0, lr=LR)

# print(f"\tUpdated z shape: {z_new.shape}")
# print("\tAux shapes:")
# sbdr.print_pytree_shapes(aux)

dz_avg = aux["dz"].mean(axis=0)


"""----------------------"""
"""Perform updates"""
"""----------------------"""

print("\nPerforming updates")
z = z0.copy()
for step in tqdm(range(STEPS), disable=True):
    key, _ = jax.random.split(key)
    z, aux = update_samples(z, lr=LR)

    if step % 10 == 0:
        print(f"\tStep: {step} \tLoss: {aux['loss']} \tMI: {aux['mi'].mean()}")
        print(f"\tact_avg = {aux['z'].mean()}")

print(f"\tUpdated z shape: {z.shape}")
print("\tAux shapes:")
sbdr.print_pytree_shapes(aux)

# print average activity in trained samples

avg_z = encode(z).mean()
print(f"\tAverage activity: {avg_z}")

# exit()


"""----------------------"""
"""Plot samples"""
"""----------------------"""

z0_plot = encode(z0)
z_plot = encode(z)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Initial Samples",
        "Trained Samples",
    ),
    horizontal_spacing=0.15,
    # vertical_spacing=0.15,
    shared_yaxes=True,
    shared_xaxes=True,
)

# plot as heatmap

fig.add_trace(
    go.Heatmap(
        z=z0_plot,
        showscale=True,
        zmin=0.0,
        zmax=1.0,
        colorscale="plasma",
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Heatmap(
        z=z_plot,
        showscale=False,
        zmin=0.0,
        zmax=1.0,
        colorscale="plasma",
    ),
    row=1, col=2,
)

fig.show()


exit()
"""----------------------"""
"""Plot distribution of unit and sample activity """
"""----------------------"""
z_encoded = encode(z)
per_sample_avg_z = z_encoded.mean(axis=-1)
per_unit_avg_z = z_encoded.mean(axis=0)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Average Unit Activity",
        "Average Sample Activity",
    ),
    horizontal_spacing=0.15,
    # vertical_spacing=0.15,
    # shared_yaxes=True,
    # shared_xaxes=True,
)

fig.add_trace(
    go.Histogram(
        x=per_unit_avg_z,
        name="Average unit activity",
        opacity=0.5,
        nbinsx=50,
        histnorm="probability",
        # marker_color=models[k][kk]["color"],
        legend="legend1"
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Histogram(
        x=per_sample_avg_z,
        name="Average sample activity",
        opacity=0.5,
        nbinsx=50,
        histnorm="probability",
        # marker_color=models[k][kk]["color"],
        legend="legend2"
    ),
    row=1, col=2,
)

fig.show()