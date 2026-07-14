import os
import jax
import jax.numpy as np
from jax import grad, vmap, jit
from pprint import pprint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from tqdm import tqdm
import infomax_sbdr as sbdr

np.set_printoptions(precision=5, suppress=True)
pio.renderers.default = "browser"

SEED = 1234

eps = 1e-1

def forward(params, x):
    # return 0.5 * (1 + np.sin(x @ params["W"] + params["b"]))
    # return np.abs(x @ params["W"] + params["b"])
    a = x @ params["W"] + params["b"]
    # v = np.clip(a, a_min=0, a_max=1)
    # v_zero = a - jax.lax.stop_gradient(a)
    # extra_grad = -v_zero * 0.01
    # v = v + (0.5 - a) * 0.01
    v = sbdr.directional_clip(a, lo=0.0, hi=1.0)
    # v = jax.nn.sigmoid(a)
    return v

def forward_avg(params, x):
    a = x @ params["W"] + params["b"]
    return a # np.clip(a, a_min=0, a_max=1)
    # return sbdr.ut.threshold_softpp(a, a_max=0.95, a_min=0.05)

def act_grad(params, x):
    a = x @ params["W"] + params["b"]
    f_a = forward(params, x)
    # 
    g = np.ones(a.shape)
    g = np.where(f_a < 0.01, 0.01, g)
    g = np.where(f_a > 0.99, -0.01, g)
    return g
    # # # sigmoid
    # # return jax.nn.sigmoid(a) * (1 - jax.nn.sigmoid(a))
    # # # Surrogate gradient without saturation
    # # return 1 / (1 + 0.1 * (a**2))

def L(params, x):
    x = x.reshape((-1, x.shape[-1]))
    
    y = forward(params, x)
    y_avg = y.mean(0)

    # compute infonce
    p_ii = (y*y).sum(-1) + eps
    p_avg = (y*y_avg).sum(-1) + eps
    loss_val = np.log(p_ii / p_avg).mean()

    return loss_val

# function with the gradient computed by hand, we want to check if it is correct
def grad_analytical_approx(params, x, true_yavg=False):
    x = x.reshape((-1, x.shape[-1]))
    x_avg = x.mean(0)

    y = forward(params, x)
    y_avg = forward(params, x_avg)
    y_avg_true = y.mean(0)

    n = (y*y).sum(-1) + eps
    d = (y*y_avg).sum(-1) + eps
    d_true = (y*y_avg_true).sum(-1) + eps

    # compute the parts of the gradients
    dl_dy = (2 * y / n[..., None] - y_avg / d[..., None])
    dl_dy_true = (2 * y / n[..., None] - y_avg_true / d_true[..., None])
    a_grad = act_grad(params, x)
    dy_dw = x[..., :, None] * a_grad[..., None, :]
    dy_db = a_grad
    # compute also for \bar{y}
    dl_dbary = - y / d[..., None]
    dl_dbary_true = - y / d_true[..., None]
    a_avg_grad = act_grad(params, x_avg)
    dbary_dw = x_avg[..., :, None] * a_avg_grad[..., None, :]
    dbary_db = a_avg_grad

    # compute the final gradient
    if true_yavg:
        dl_dw = (
            dl_dy_true[..., None, :] * dy_dw 
            +
            dl_dbary_true[..., None, :] * dbary_dw
        )
        dl_db = (
            dl_dy_true * dy_db
            +
            dl_dbary_true * dbary_db
        )
    else:
        dl_dw = (
            dl_dy[..., None, :] * dy_dw 
            +
            dl_dbary[..., None, :] * dbary_dw
        )
        dl_db = (
            dl_dy * dy_db
            +
            dl_dbary * dbary_db
        )
    
    # use a homeostatic gradient on b instead
    # dl_db = dl_db + (0.02 - y)
    return {
        "W": dl_dw.mean(0),
        "b": dl_db.mean(0),
    }

key = jax.random.key(SEED)

# # POSITIVE init params
# key, subkey = jax.random.split(key, 2)
# params = {
#     "W": np.abs(jax.random.normal(key, (IN_FEATURES, OUT_FEATURES))),
#     "b": np.abs(jax.random.normal(subkey, (OUT_FEATURES,))),
# }
# # init samples
# key, _ = jax.random.split(key, 2)
# x = np.abs(jax.random.normal(key, (BATCH_SIZE, IN_FEATURES)))
BATCH_SIZE = 1024
IN_FEATURES = 89
OUT_FEATURES = 128

# REAL init params
key, subkey = jax.random.split(key, 2)
params = {
    "W": 0.05 * jax.random.normal(key, (IN_FEATURES, OUT_FEATURES)),
    "b": np.zeros((OUT_FEATURES,)), # jax.random.normal(subkey, (OUT_FEATURES,)),
}
# init samples
key, _ = jax.random.split(key, 2)
x = jax.random.normal(key, (BATCH_SIZE, IN_FEATURES))

# Compute gradients
g_an = grad_analytical_approx(params, x)
g_auto = grad(L)(params, x)


# Compare
diff = jax.tree.map(lambda x, y: x - y, g_an, g_auto)

# pprint(diff)

# print some nice diff statistics, including quantiles, mean, std
qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
print(f"Quantiles: {np.quantile(diff['W'], qs)}")
print(f"Mean: {np.mean(diff['W'])}")
print(f"Std: {np.std(diff['W'])}")

print(f"Quantiles: {np.quantile(diff['b'], qs)}")
print(f"Mean: {np.mean(diff['b'])}")
print(f"Std: {np.std(diff['b'])}")

"""----------------"""
""" Run updates """
"""----------------"""

N_STEPS = 15000
ETA = 0.05
# updates the params iteratively using the gradient
# like sgd (ascent, actually), to compare what changes between using the 
# analytically approximate gradient and the actual gradient 
# computed automatically on the optimality function

print("Running updates")

params_original = params.copy()
params_an = params.copy()
params_auto = params.copy()

@jit
def update_params_analytical(params, x):
    g = grad_analytical_approx(params, x, true_yavg=True)
    return {
        "W": params["W"] + ETA * g["W"],
        "b": params["b"] + ETA * g["b"],
    }

@jit
def update_params_automatic(params, x):
    g = grad(L)(params, x)
    new_params = {
        "W": params["W"] + ETA * g["W"],
        "b": params["b"] + ETA * g["b"],
    }
    # # normalize W to be unit vectors in the input space
    # new_params["W"] = new_params["W"] / np.linalg.norm(new_params["W"], axis=0)
    return new_params


for i in tqdm(range(N_STEPS)):
    params_an = update_params_analytical(params_an, x)
    params_auto = update_params_automatic(params_auto, x)

# Compute encodings with trained weights
y_original = forward(params_original, x)
y_an = forward(params_an, x)
y_auto = forward(params_auto, x)

# Compute losses
loss_original = L(params_original, x)
loss_an = L(params_an, x)
loss_auto = L(params_auto, x)

print(f"Losses:")
print(f"\tLoss (original): {loss_original}")
print(f"\tLoss (analytical): {loss_an}")
print(f"\tLoss (automatic): {loss_auto}")

# print statistics on sparsity
print(f"Sparsity (original): {y_original.mean()}")
print(f"Sparsity (analytical): {y_an.mean()}")
print(f"Sparsity (automatic): {y_auto.mean()}")

"""----------------"""
""" Plot params and activations """
"""----------------"""

# Weight heatmap, 4 subplots, initial, analytical, automatic, diff

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Initial Weights",
        "Analytical Weights",
        "Automatic Weights",
        "Diff",
    ),
    horizontal_spacing=0.15,
    # vertical_spacing=0.15,
    shared_yaxes=True,
    shared_xaxes=True,
)

fig.add_trace(
    go.Heatmap(z=params["W"]),
    row=1, col=1,
)

fig.add_trace(
    go.Heatmap(z=params_an["W"]),
    row=1, col=2,
)

fig.add_trace(
    go.Heatmap(z=params_auto["W"]),
    row=2, col=1,
)

fig.add_trace(
    go.Heatmap(z=params_auto["W"] - params_an["W"]),
    row=2, col=2,
)

fig.update_layout(height=600, width=800)
fig.show()

# Activation heatmap, 4 subplots, initial, analytical, automatic, diff

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Initial Activations",
        "Analytical Activations",
        "Automatic Activations",
        "Diff",
    ),
    horizontal_spacing=0.15,
    # vertical_spacing=0.15,
    shared_yaxes=True,
    shared_xaxes=True,
)

fig.add_trace(
    go.Heatmap(
        z=y_original,
        zmin=0.0,
        zmax=1.0,
        colorscale="viridis",
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Heatmap(
        z=y_an,
        zmin=0.0,
        zmax=1.0,
        colorscale="viridis",
    ),
    row=1, col=2,
)

fig.add_trace(
    go.Heatmap(
        z=y_auto,
        zmin=0.0,
        zmax=1.0,
        colorscale="viridis",
    ),
    row=2, col=1,
)

fig.add_trace(
    go.Heatmap(
        z=y_auto - y_an,
        colorscale="RdBu",
        zmin=-1.0,
        zmax=1.0,
    ),
    row=2, col=2,
)

# fig.update_layout(height=600, width=800)
fig.show()