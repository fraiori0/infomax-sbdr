import numpy as np
import plotly.graph_objects as go
import sklearn as skl
from sklearn.decomposition import PCA

SEED = 9876

N_IN_FEATURES = 256
N_OUT_FEATURES = 64
N_SAMPLES = 1000

N_GAUSSIANS = 300

rng = np.random.default_rng(SEED)

# Generate samples as a mixture of Gaussians
mus = rng.uniform(-1.0, 1.0, size=(N_GAUSSIANS, N_IN_FEATURES))
stds = rng.uniform(0.1, 0.5, size=(N_GAUSSIANS, N_IN_FEATURES))
# Sample from each gaussian, fpr each sample
gaussian_samples = rng.normal(mus, stds, size=(N_SAMPLES, N_GAUSSIANS,N_IN_FEATURES))
# random mask selecting one gaussian per sample
mask = rng.choice(N_GAUSSIANS, size=(N_SAMPLES,))
samples = gaussian_samples[np.arange(N_SAMPLES), mask]


# Perform PCA decomposition
pca = PCA(n_components=N_OUT_FEATURES, random_state=SEED)
pca.fit(samples)
# Take the principal vector
pcs = pca.components_
print(pcs.shape)

# Randomly change the sign of each principal vector
signs = rng.choice([-1, 1], size=(N_OUT_FEATURES,))
pcs_rndsign = pcs * signs[..., None]

# Perform projection on both
samples_pca = pca.transform(samples)
samples_pca_rndsign = (samples[..., None, :] * pcs_rndsign).mean(axis=-1)

print(samples_pca.shape)
print(samples_pca_rndsign.shape)

# Hard thresholding with Heaviside step function
xs = (samples_pca > 0).astype(np.float64)
xs_rndsign = (samples_pca_rndsign > 0).astype(np.float64)

# Plot covariance matrix of binarized data

covs = np.cov(xs.T)
covs_rndsign = np.cov(xs_rndsign.T)

fig = go.Figure(data=go.Heatmap(z=covs))
fig.show()

fig = go.Figure(data=go.Heatmap(z=covs_rndsign))
fig.show()

