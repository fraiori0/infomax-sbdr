import os
import sys
import jax
import jax.numpy as np
from jax import grad, jit, vmap
import infomax_sbdr as sbdr
from time import time
import colorsys
from functools import partial


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


def gen_ys_k_active_weighted(
    key, ws: np.ndarray, n_active_features: int, n_samples: int
):
    """Generate random binary samples with exactly n_active_features active units each/
    Each unit will have a value of either 0 or 1.

    The weights can be used to sample from a weighted distribution.

    Args:
        key: jax random key
        n_active_features: number of active features
        n_samples: number of samples to be generated
        ws: array of shape (n_tot_features,)

    Returns:
        ys: array of shape (n_samples, n_tot_features)
    """

    assert n_active_features <= ws.shape[-1]
    assert n_active_features > 0

    one_hot_matrix = np.eye(ws.shape[-1], dtype=bool)
    # for each sample we will select n_active_features rows of the one_hot_matrix
    idxs = np.arange(ws.shape[-1])
    keys = jax.random.split(key, n_samples)
    # select n_active_features indexes using weights
    idxs = vmap(
        partial(
            jax.random.choice,
            replace=False,
            p=ws,
        ),
        in_axes=(0, None, None),
        out_axes=0,
    )(keys, idxs, (n_active_features,))

    # select the one_hot_matrix elements
    ys = np.take(one_hot_matrix, idxs, axis=0)
    # # sum the one_hot_matrix elements selected for each sample
    ys = ys.sum(axis=-2)

    return ys


def get_beta_dist_mean_std(alpha, beta):
    """
    Get the mean and std parameters for a Beta distribution with the given alpha and beta parameters.

    Note, this function is compatible with broadcastable mean and variance arrays.
    """

    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    std = np.sqrt(variance)

    return mean, std


def get_beta_dist_alpha_beta(mean, concentration):
    """
    Get the alpha and beta parameters for a Beta distribution with the given mean and concentration.

    Note, this function is compatible with broadcastable mean and concentration arrays.
    """

    # Calculate Beta distribution parameters based on mean and concentration
    # alpha + beta = concentration
    # alpha / (alpha + beta) = mean
    alpha = concentration * mean
    beta = concentration * (1.0 - mean)

    return alpha, beta


def generate_color(h, l, s):
    """Convert HLS values (range [0,1]) to an rgb string for Plotly."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def generate_ps_beta_distribution(
    key: jax.random.key,
    n_total_features: int,
    n_mean_active: float,
    concentration: float,
    n_batch_size: int,
) -> np.ndarray:
    """
    Generates probabilities for a multivariate Bernoulli distribution.

    The probabilities are sampled from a Beta distribution such that their
    mean value is controlled, and the spread (variance) among them is
    controlled by the concentration parameter. The expected sum of the
    generated probabilities will be equal to n_mean_active.

    Args:
        key: JAX PRNG key.
        n_total_features: The total number of boolean features (dimensionality).
        n_mean_active: The desired average number of active features per sample.
                       This corresponds to the expected sum of the generated
                       probabilities. Must satisfy 0 < n_mean_active < n_total_features.
        concentration: Controls the spread of probabilities.
                       Higher values lead to lower spread (probabilities closer
                       to the mean). Lower values lead to higher spread.
                       Must be positive.
        n_batch_size: The number of different samples to generate, each of (n_total_features,)

    Returns:
        A JAX array of shape (n_batch_size, n_total_features) containing the probabilities,
        where each element on the last axis p_i is the probability of the i-th feature being active.
    """
    if not (0 < n_mean_active < n_total_features):
        raise ValueError(
            f"n_mean_active ({n_mean_active}) must be strictly between 0 and n_total_features ({n_total_features})"
        )
    if not concentration > 0:
        raise ValueError(f"concentration ({concentration}) must be positive.")

    # Calculate the target mean probability for each feature
    mean_p = n_mean_active / n_total_features

    alpha, beta = get_beta_dist_alpha_beta(mean_p, concentration)

    # Ensure alpha and beta are positive (handled by input validation)
    # JAX's beta function requires positive parameters.

    # Sample probabilities from the Beta distribution
    ps = jax.random.beta(key, alpha, beta, shape=(n_batch_size, n_total_features))

    # Clip probabilities to avoid potential numerical issues at exact 0 or 1
    # although Beta distribution naturally stays within (0, 1) for alpha, beta > 0
    ps = np.clip(ps, np.finfo(ps.dtype).eps, 1.0 - np.finfo(ps.dtype).eps)

    return ps, (alpha, beta)
