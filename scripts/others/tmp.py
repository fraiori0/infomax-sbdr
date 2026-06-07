import jax
from jax import vmap, grad, jit
import jax.numpy as np
import plotly.graph_objects as go
import sklearn as skl
from sklearn.decomposition import PCA
import infomax_sbdr as sbdr

np.set_printoptions(precision=4, suppress=True)

SEED = 0

features = 128
in_features = 40
k_size = 5


mask = sbdr.inits.time_conv_boolean_mask()(
    key=jax.random.PRNGKey(SEED),
    shape=(k_size, in_features, features),
    axis=0,
)

print(mask)
print(mask.shape)

print(mask.sum(axis=0))

print(mask.sum(axis=(1, 2)))

print(mask.sum())