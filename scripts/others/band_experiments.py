import os
import jax
import jax.numpy as np
import numpy as onp
import plotly.graph_objects as go
import sklearn as skl
from sklearn.decomposition import PCA
import infomax_sbdr as sbdr
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)

SEED = 987156

N_IN_FEATURES = 3
N_SAMPLES = 200

N_GAUSSIANS = 7

rng = onp.random.default_rng(SEED)

# Generate samples as a mixture of Gaussians
mus = rng.uniform(-1.0, 1.0, size=(N_GAUSSIANS, N_IN_FEATURES))
stds = rng.uniform(0.1, 0.5, size=(N_GAUSSIANS, N_IN_FEATURES))
# Sample from each gaussian, for each sample
gaussian_samples = rng.normal(mus, stds, size=(N_SAMPLES, N_GAUSSIANS,N_IN_FEATURES))
# random mask selecting one gaussian per sample
mask = rng.choice(N_GAUSSIANS, size=(N_SAMPLES,))
samples = gaussian_samples[onp.arange(N_SAMPLES), mask]


# Do the same to generate two cluster of negative samples
N_SAMPLES_NEG = 50
N_GAUSSIANS_NEG = 2
mus_neg = onp.clip(5*rng.uniform(-1.0, 1.0, size=(N_GAUSSIANS_NEG, N_IN_FEATURES)), -1.5, 1.5)
stds_neg = rng.uniform(0.1, 0.2, size=(N_GAUSSIANS_NEG, N_IN_FEATURES))
# Sample from each gaussian, for each sample
gaussian_samples_neg = rng.normal(mus_neg, stds_neg, size=(N_SAMPLES_NEG, N_GAUSSIANS_NEG, N_IN_FEATURES))
# random mask selecting one gaussian per sample
mask_neg = rng.choice(N_GAUSSIANS_NEG, size=(N_SAMPLES_NEG,))
samples_neg = gaussian_samples[onp.arange(N_SAMPLES_NEG), mask_neg]


# Initialize an hyperplane

params = {
    "w": rng.standard_normal(N_IN_FEATURES),
    "b": onp.zeros((1,))
}
params["w"] = params["w"] / (np.linalg.norm(params["w"]) + 1e-6)
params_original = params.copy()

def get_plane_points(params, x_range=(-1, 1), y_range=(-1, 1), resolution=5):

    # Create meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = -(params["w"][0] * X + params["w"][1] * Y + params["b"]) / (params["w"][2] + 1e-6)
    return np.stack([X, Y, Z], axis=-1)
    
print("\nOriginal shape of parameters:")
sbdr.print_pytree_shapes(params)

error = np.dot(samples, params["w"]) + params["b"]
error_neg = np.dot(samples_neg, params["w"]) + params["b"]

print(f"\tError before training: {error.mean()} ({error.std()})")
print(f"\t    negative samples: {error_neg.mean()} ({error_neg.std()})")

# Train the weights using sgd

N_EPOCHS = 75
N_BATCH = 10
LR = 0.1
LR_NEG = 0.1
REG = 0

history = {
    "norm_w": [],
}

for ne in tqdm(range(N_EPOCHS)):
    
    # shuffle the samples
    samples = samples[rng.permutation(N_SAMPLES)]
    # shuffle also the negative samples
    samples_neg = samples_neg[rng.permutation(N_SAMPLES_NEG)]
    
    # pass through batches (skip last)
    for nb in tqdm(range(N_SAMPLES // N_BATCH), leave=False):
        
        x_batch = samples[nb * N_BATCH : (nb + 1) * N_BATCH]
        # take N_BATCH negative samples at random
        x_batch_neg = samples_neg[rng.choice(N_SAMPLES_NEG, size=N_BATCH)]

        # compute projection on hyperplane
        y_batch = np.dot(x_batch, params["w"]) + params["b"]
        y_batch_neg = np.dot(x_batch_neg, params["w"]) + params["b"]
        # just take the signs
        y_batch = np.sign(y_batch)
        y_batch_neg = np.sign(y_batch_neg)

        # # # Compute parameter updates FOR POSITIVE SAMPLES
        rho = np.dot(params["w"], params["w"])
        # Weights
        # dw = y_batch[..., None] * params["w"] - rho * x_batch - REG*rho*params["w"]
        dw = -y_batch[..., None] * x_batch - REG*params["w"]
        dw = dw.mean(axis=0)
        # Bias
        db = -y_batch[..., None]
        db = db.mean(axis=0)
        # # # Update with positive samples
        params["w"] = params["w"] + LR * dw
        params["b"] = params["b"] + LR * db

        # # # Compute parameter updates FOR NEGATIVE SAMPLES
        rho = np.dot(params["w"], params["w"])
        # Weights
        # dw = rho * x_batch_neg - y_batch_neg[..., None] * params["w"] #  - REG*rho*params["w"]
        dw = y_batch_neg[..., None] * x_batch_neg - REG*params["w"]
        dw = dw.mean(axis=0)
        # # Bias
        # db = -y_batch_neg[..., None]
        # db = db.mean(axis=0)
        # # # Update with negative samples
        params["w"] = params["w"] + LR_NEG * dw
        # params["b"] = params["b"] + LR * db

        # Store the norm of W
        history["norm_w"].append(np.linalg.norm(params["w"]))

        # # # Normalization
        # # Unit norm (standard, L2)
        # params["w"] = params["w"] / (np.linalg.norm(params["w"]) + 1e-5)
        # # Unit L1 norm
        # params["w"] = params["w"] / np.abs(params["w"]).sum(axis=-1, keepdims=True)

    

print("\nFinal shape of parameters:")
sbdr.print_pytree_shapes(params)
error = np.dot(samples, params["w"]) + params["b"]
error_neg = np.dot(samples_neg, params["w"]) + params["b"]

print(f"\tError after training: {error.mean()} ({error.std()})")
print(f"\t    negative samples: {error_neg.mean()} ({error_neg.std()})")



# Plot points in 3D

fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=samples[:, 0],
        y=samples[:, 1],
        z=samples[:, 2],
        mode="markers",
        marker=dict(size=2),
    )
)

fig.add_trace(
    go.Scatter3d(
        x=samples_neg[:, 0],
        y=samples_neg[:, 1],
        z=samples_neg[:, 2],
        mode="markers",
        marker=dict(size=2),
    )
)


# Plot both the original and the trained hyperplane as go.Surface
points_original = get_plane_points(params_original)
points_trained = get_plane_points(params)

fig.add_trace(
    go.Surface(
        x=points_original[:, :, 0],
        y=points_original[:, :, 1],
        z=points_original[:, :, 2],
        opacity=0.2,
        colorscale=[[0, 'blue'], [1.0, 'blue']],
    )
)

fig.add_trace(
    go.Surface(
        x=points_trained[:, :, 0],
        y=points_trained[:, :, 1],
        z=points_trained[:, :, 2],
        opacity=0.5,
        colorscale=[[0, 'green'], [1.0, 'green']],
    )
)

# set all axis to have the same scale, so we can get a geometric sense of what happened
fig.update_layout(scene_aspectmode="data") # , scene_aspectratio=dict(x=1, y=1, z=1))

fig.show()


# Plot the norm of w over time
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=np.arange(len(history["norm_w"])),
        y=history["norm_w"],
        mode="lines",
    )
)

fig.show()