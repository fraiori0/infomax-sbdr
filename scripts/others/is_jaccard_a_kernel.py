import jax
import jax.numpy as np
from jax import jit, grad, vmap
import infomax_sbdr as sbdr
from functools import partial
from time import time
import numpy as onp

"""
Check here https://stats.stackexchange.com/questions/48506/what-function-could-be-a-kernel
"""


np.set_printoptions(precision=4, suppress=True)

EPS = 0.0
N_SAMPLES = 200
N_DIMS = 5
N_RESAMPLES = 200


def gen_ys_k_active(key, n_active_features: int, n_samples: int, n_tot_features: int):
    """Generate random binary samples with exactly n_active_features active units each/
    Each unit will have a value of either 0 or 1.

    Args:
        key: jax random key
        n_active_features: number of active features
        n_samples: number of samples to be generated
        n_tot_features: total number of features (n_active_features will be set to 1, the rest to 0)

    Returns:
        ys: array of shape (n_samples, n_tot_features)
    """

    assert n_active_features <= n_tot_features
    assert n_active_features > 0

    one_hot_matrix = np.eye(n_tot_features, dtype=bool)
    # select n_samples samples from the one_hot_matrix
    idxs = np.arange(n_tot_features)
    idxs = np.stack([idxs] * n_samples, axis=0)
    # shuffle index axis independently
    idxs = jax.random.permutation(key, idxs, axis=-1, independent=True)
    # take only the first n_k indexes
    idxs = idxs[..., :n_active_features]

    # select the one_hot_matrix elements
    ys = np.take(one_hot_matrix, idxs, axis=0)
    # # sum the one_hot_matrix elements selected for each sample
    ys = ys.sum(axis=-2)

    return ys


sim_fn = partial(
    sbdr.config_similarity_dict["jaccard"],
    eps=EPS,
)


eig_min = []
key = jax.random.key(int(time()))
for i in range(N_RESAMPLES):

    key, _ = jax.random.split(key)

    xs = jax.random.uniform(key, shape=(N_SAMPLES, N_DIMS), dtype=np.float64)

    # key, _ = jax.random.split(key)
    # xs = jax.random.bernoulli(key, xs)
    # xs = gen_ys_k_active(key, 1, N_SAMPLES, N_DIMS)

    K = onp.array(sim_fn(xs[:, None, :], xs[None, :, :]))

    # compute eigenvalues
    eigenvalues, _ = onp.linalg.eig(K)

    eig_min.append(eigenvalues.min())


# convert to numpy array
eig_min = onp.array(eig_min)
# print the minimum of the minimum eigenvalues
print(f"\tMinimum eigenvalue: {eig_min.min()}")
