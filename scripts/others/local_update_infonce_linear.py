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
N_FEATURES = 128
N_SAMPLES = 2000

EPS = 1e-1
P_DES = 0.05
LR = 0.1
STEPS = 2000
ALPHA = 0.0

GAMMA = 0.9
# GAMMA = jax.random.uniform(jax.random.key(SEED), shape=(N_FEATURES,), minval=0.3, maxval=0.9)

RANGE = (0.0, 1.0)
K_WINNERS = int(N_FEATURES * P_DES)


key = jax.random.PRNGKey(SEED)


def clip_ste(x):
    x_zero = x - jax.lax.stop_gradient(x)
    zero = np.where(
        np.logical_or(x < RANGE[0], x > RANGE[1]),
        0.05*x_zero,
        x_zero,
    )
    val = np.clip(x, a_min=RANGE[0], a_max=RANGE[1])
    return jax.lax.stop_gradient(val) + zero

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
    # # Encode with sigmoid
    # return sbdr.sigmoid_ste(z)
    # return jax.nn.sigmoid(z)

    # Compute k-winner-takes-all
    _, idx_winners = jax.lax.top_k(z, K_WINNERS)

    f = np.eye(z.shape[-1])[idx_winners]
    f = f.sum(-2)
    return f



def expsim(z1, z2, eps=1e-8):
    return (z1*z2).sum(-1)

# def infonce(z, z_avg, eps=1e-8):
#     return np.log(expsim(z, z_avg) + eps) - np.log(expsim(z, z) + eps)

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
    pii = (z*z).sum(-1) + eps
    pij = (z[..., None, :] * z[..., :, None]).sum(-1) + eps
    mi = sbdr.infonce(pii, pij, eps=1e-6)
    loss_val = -mi
    return loss_val

def infonce_avg(z, eps=1e-8):
    z_avg = z.reshape((-1, z.shape[-1])).mean(axis=0)
    pii = (z*z).sum(-1) + eps
    p_avg = (z * z_avg).sum(-1) + eps
    mi = np.log(pii/p_avg)
    loss_val = -mi
    return loss_val

@jit
def update_samples(z, lr):
    # z of shape (*batch_dims, features)
    # update each z according to it's gradient
    def loss(z):
        z_binary = encode(z)
        # z_coded = clip_ste(z)
        mi = infonce_avg(z, eps=EPS)
        loss_val = mi.sum()
        mi = mi.mean()
        mi_binary = infonce_avg(z_binary, eps=EPS).mean()

        aux = {
            "z": z,
            "z_binary": z_binary,
            "mi": mi,
            "mi_binary": mi_binary
        }
        return loss_val, aux

    z_shape = z.shape
    z = z.reshape((-1, z.shape[-1]))

    (loss_val, aux), dz = jax.value_and_grad(loss, has_aux=True)(z)

    z_new = z - lr * dz
    z_new = z_new.reshape(z_shape)

    # clip in range
    z_new = np.clip(z_new, RANGE[0], RANGE[1])

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
auxes = {k:[] for k in aux.keys()}
z = z0.copy()
for step in tqdm(range(STEPS), disable=True):
    key, _ = jax.random.split(key)
    z, aux = update_samples(z, lr=LR)

    for k in aux.keys():
        auxes[k].append(aux[k])

    if step % 10 == 0:
        print(f"\tStep: {step} \tLoss: {aux['loss']} \tMI: {aux['mi_binary'].mean()}")
        print(f"\tact_avg = {aux['z'].mean()}")

# convert to numpy arrays
for k in auxes.keys():
    auxes[k] = np.array(auxes[k])

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

z0_plot = z0
z_plot = z
z_binary_plot = encode(z)
z_scale = (float(min(z0_plot.min(), z_plot.min())), float(max(z0_plot.max(), z_plot.max())))

# order units in order of activity
unit_avg_activity = z_plot.mean(axis=0)
unit_order = np.argsort(unit_avg_activity)
z0_plot = z0_plot[:, unit_order]
z_plot = z_plot[:, unit_order]
z_binary_plot = z_binary_plot[:, unit_order]

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "Initial Samples",
        "Trained Samples",
        "Binarized Samples",
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
        zmin=z_scale[0],
        zmax=z_scale[1],
        colorscale="plasma",
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Heatmap(
        z=z_plot,
        showscale=False,
        zmin=z_scale[0],
        zmax=z_scale[1],
        colorscale="plasma",
    ),
    row=1, col=2,
)

fig.add_trace(
    go.Heatmap(
        z=z_binary_plot,
        showscale=False,
        zmin=0.0,
        zmax=1.0,
        colorscale="plasma",
    ),
    row=1, col=3,
)

fig.show()

"""----------------------"""
"""Plot training history (mi) """
"""----------------------"""

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=np.arange(len(auxes["mi_binary"])),
        y=auxes["mi_binary"],
        mode="lines",
        name="mi_binary",
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(len(auxes["mi"])),
        y=auxes["mi"],
        mode="lines",
        name="mi",
    )
)

fig.update_layout(
    title="Training History",
    xaxis_title="Step",
    yaxis_title="MI",
)

fig.show()



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