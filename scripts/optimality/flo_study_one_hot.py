import os
import sys
import jax
import jax.numpy as np
from jax import grad, jit, vmap
import sbdr
import plotly.graph_objects as go
from jax.experimental import sparse
from time import time
import argparse
from tqdm import tqdm


# Create an argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "-s",
    "--seed",
    type=str,
    help="Path to the folder containing the files for the validation set",
    default=None,
)
parser.add_argument(
    "-n", "--name", type=str, help="Name of this Optuna study", default="test"
)


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


def gen_zs_normal(
    key,
    n_nonzero_features: int,
    n_samples: int,
    n_tot_features: int,
    mu: float = 1.0,
    sigma: float = 1.0,
):
    """Generate n_samples different probability distributions over the activity of n_tot_features.

    Each distribution will be masked so that only n_active_features have a non-zero probability of activation.

    Each distribution is obtained by drawing from a normal distribution and passing through a sigmoid to obtain a per-feature activation probability
    bounded in [0, 1].

    Args:
        key: jax random key
        n_nonzero_features: number of features with non-zero probability of activation
        n_samples: number of samples to be generated
        n_tot_features: total number of features
        mu: mean of the normal distribution
        sigma: standard deviation of the normal distribution

    Returns:
        ys: array of shape (n_samples, n_tot_features)
    """
    assert n_nonzero_features <= n_tot_features
    assert n_nonzero_features >= 0

    key_normal, key_mask = jax.random.split(key, 2)

    # Generate the per-feature probability of activation from a normal distribution
    ps = jax.random.normal(key_normal, shape=(n_samples, n_tot_features))
    ps = ps * sigma + mu

    # # Apply the sigmoid
    # ps = jax.nn.sigmoid(ps)
    ps = np.clip(ps, 1e-3, 1.0)

    # Mask the probabilities so that only n_nonzero_features have a non-zero probability of activation
    ps_mask = gen_ys_k_active(key_mask, n_nonzero_features, n_samples, n_tot_features)
    ps = ps * ps_mask

    return ps


if __name__ == "__main__":

    SEED = int(time())

    key = jax.random.PRNGKey(SEED)

    N_TOT_FEATURES = 128
    N_SAMPLES = 1000
    N_NONZERO_FEATURES = [1, 2, 3, 5, 7]  #
    # combinations of mean and variance of the normal distribution
    # to be tested
    MUS = [2.0, 1.0, 0.5, 0.0]
    SIGMAS = [1.0, 0.7, 0.5, 0.3, 0.1]

    MIs = []

    for n_nonzero_features in tqdm(N_NONZERO_FEATURES):

        MIs.append([])

        for mu in tqdm(MUS, leave=False):

            MIs[-1].append([])

            for sigma in tqdm(SIGMAS, leave=False):

                key, _ = jax.random.split(key)

                # Generate samples
                zs = gen_zs_normal(
                    key,
                    n_nonzero_features,
                    N_SAMPLES,
                    N_TOT_FEATURES,
                    mu=mu,
                    sigma=sigma,
                )

                # print(zs.sum(axis=-1))
                # print(zs.sum(axis=-1).mean())
                # print(zs.sum(axis=-1).std())
                # exit()

                sim_fn = sbdr.gamma_similarity
                p_ii = sim_fn(zs, zs)
                p_ij = sim_fn(zs[:, None, :], zs[None, :, :])

                # print(p_ij.mean(axis=-1))
                # print(p_ij.std(axis=-1))
                # print(p_ii.mean())
                # print(p_ii.std())
                # exit()

                # print(p_ii.shape)
                # print(p_ij.shape)

                # Compute InfoNCE bound
                # evaluate the term inside the log, for each distribution
                # (it is practically an estimate of the pointwise mutual information)
                pmis = np.log(p_ii / (p_ij.mean(axis=-1) + 1e-6) + 1e-6)
                mi = pmis.mean(axis=-1)

                MIs[-1][-1].append(mi)

    MIs = np.array(MIs)

    print(MIs.shape)

    # plot the curves of MI
    # use different curves for the amount of non-zero features and for different variances
    # x axis should be the means
    # y axis should be the MI
    # group together in the legend the non-zero features

    fig = go.Figure()

    max_y = 0

    for i, nonzero_feature in enumerate(N_NONZERO_FEATURES):
        for j, sigma in enumerate(SIGMAS):
            fig.add_trace(
                go.Scatter(
                    x=MUS,
                    y=MIs[i, :, j],
                    mode="lines+markers",
                    name=f"{nonzero_feature} features, {sigma} sigma",
                    # line properties
                    line=dict(width=2),
                    # marker properties
                    marker=dict(size=6),
                    legendgroup=f"{nonzero_feature} features",
                )
            )

            max_y = max(max_y, MIs[i, :, j].max())

    fig.update_layout(
        title=f"MI vs Added Noise - {N_TOT_FEATURES} Features",
        xaxis_title="Noise level (mean)",
        yaxis_title="MI",
        # legend_title="Baseline active features<br>(no noise)",
        # set fig style to white
        plot_bgcolor="white",
        paper_bgcolor="white",
        # and grey gridlines
        xaxis=dict(gridcolor="lightgrey"),
        yaxis=dict(gridcolor="lightgrey", range=[0, max_y + 1]),
        # tick every unit on the x axis
        xaxis_dtick=1,
        # and every 0.5 units on the y axis
        yaxis_dtick=0.5,
    )

    fig.show()
