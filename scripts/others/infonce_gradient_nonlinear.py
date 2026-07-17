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
    a = x @ params["W"] + params["b"]
    # v = sbdr.directional_clip_rev(a, lo=0.0, hi=1.0)
    # v = 2*jax.nn.sigmoid(a)
    # v = np.sqrt(a**2 + 1e-4)
    v = sbdr.directional_clip_rev(a, lo=0.0, hi=1.0)

    # # compute distances
    # d = x[..., None] - params["W"]
    # d = np.linalg.norm(d, axis=-2)
    # d = d + params["b"]
    # # straight-through threshold
    # v = sbdr.threshold_softgradient(-d)
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


# @jax.jit
def sample_gmm(key, weights, means, covs, num_samples):
    """
    Sample synthetic data from a Gaussian Mixture Model efficiently in JAX.
    
    Args:
        key: jax.random.PRNGKey for reproducibility.
        weights: Array of shape (K,) representing component probabilities (must sum to 1).
        means: Array of shape (K, D) representing component means.
        covs: Array of shape (K, D, D) representing component covariance matrices.
        num_samples: Integer N, the number of samples to generate.
        
    Returns:
        samples: Array of shape (N, D) containing the generated synthetic data.
        assignments: Array of shape (N,) containing the component index for each sample.
    """
    # 1. Split PRNG keys for the categorical draw and the normal noise draw
    key_cat, key_norm = jax.random.split(key)
    
    # 2. Sample component assignments for all N data points
    # jax.random.categorical expects unnormalized log-probabilities (logits)
    logits = np.log(weights)
    assignments = jax.random.categorical(key_cat, logits, shape=(num_samples,))
    
    # 3. Precompute Cholesky decomposition for all K covariance matrices simultaneously
    # Shape: (K, D, D)
    cholesky_factors = np.linalg.cholesky(covs)
    
    # 4. Generate standard normal noise for all N samples
    num_dims = means.shape[1]
    epsilon = jax.random.normal(key_norm, shape=(num_samples, num_dims))
    
    # 5. Gather the specific mean and Cholesky factor for each sample's assigned component
    # Advanced indexing gathers rows without copying on XLA
    chosen_means = means[assignments]         # Shape: (num_samples, num_dims)
    chosen_L = cholesky_factors[assignments]  # Shape: (num_samples, num_dims, num_dims)
    
    # 6. Apply affine transformation: x_n = L_n @ epsilon_n + mu_n
    # 'nij,nj->ni' performs batched matrix-vector multiplication across dimension N
    samples = np.einsum('nij,nj->ni', chosen_L, epsilon) + chosen_means
    
    return samples, assignments


"""----------------"""
""" Initialization """
"""----------------"""

key = jax.random.key(SEED)

BATCH_SIZE = 2048
IN_FEATURES = 75
OUT_FEATURES = 64
N_GAUSSIANS = 5

# # # Init Params
key, subkey = jax.random.split(key, 2)
# # REAL init params
params = {
    "W": (1/IN_FEATURES) * jax.random.normal(key, (IN_FEATURES, OUT_FEATURES)),
    "b": 0 * np.ones((OUT_FEATURES,)), # jax.random.normal(subkey, (OUT_FEATURES,)),
}
# # POSITIVE init params
# params = {
#     "W": 0.05*np.abs(jax.random.normal(key, (IN_FEATURES, OUT_FEATURES))),
#     "b": 0 * np.ones((OUT_FEATURES,)), # np.abs(jax.random.normal(subkey, (OUT_FEATURES,))),
# }

# # # Init Samples
# From multiple random gaussians
key, subkey = jax.random.split(key, 2)
mus = jax.random.normal(key, (N_GAUSSIANS, IN_FEATURES))
ws = np.ones((N_GAUSSIANS,)) / N_GAUSSIANS
# sample positive definite covariance matrices
covs = jax.random.normal(subkey, (N_GAUSSIANS, IN_FEATURES, IN_FEATURES))
covs = 0.1 * ((covs @ covs.transpose(0, 2, 1)) + np.eye(IN_FEATURES))
# Sample
key, _ = jax.random.split(key, 2)
x, _ = sample_gmm(subkey, ws, mus, covs, BATCH_SIZE)

# # # Preprocess
# Rescale each component independently to z-score
x = (x - x.mean(0)) / x.std(0)
# # Rescale min-max to 0-1
# x = (x - x.min(0)) / (x.max(0) - x.min(0))


# # # Gradients
# Compute gradients
g_an = grad_analytical_approx(params, x)
g_auto = grad(L)(params, x)
# Compare
diff = jax.tree.map(lambda x, y: x - y, g_an, g_auto)
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
ETA = 0.3
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
    params["W"] = params["W"] + ETA * g["W"]
    params["b"] = params["b"] + ETA * g["b"]
    
    # # normalize weights and zero-out bias
    # params["W"] = params["W"] / np.linalg.norm(params["W"], axis=0)
    # params["b"] = np.zeros_like(params["b"])
    return params

@jit
def update_params_automatic(params, x):
    g = grad(L)(params, x)
    params["W"] = params["W"] + ETA * g["W"]
    params["b"] = params["b"] + ETA * g["b"]
    
    # normalize weights and zero-out bias
    # params["W"] = params["W"] / (np.linalg.norm(params["W"], axis=0, keepdims=True) + 1e-5)

    # # softmax (so they are positive and sum to 1)
    # params["W"] = jax.nn.softmax(params["W"], axis=0)

    # clip bias to be non-positive
    # params["b"] = np.clip(params["b"], a_min=None, a_max=0)
    return params


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