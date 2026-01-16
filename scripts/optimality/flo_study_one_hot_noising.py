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

np.set_printoptions(precision=4, suppress=True)


def get_beta_dist_params(mean, variance):
    """
    Get the alpha and beta parameters for a beta distribution with the given mean and variance.

    Note, this function is compatible with broadcastable mean and variance arrays.
    """

    # assess the boundaries are correct
    assert np.all(mean <= 1) and np.all(mean >= 0)
    assert np.all(variance < mean - mean**2)

    tmp = (mean * (1 - mean)) / variance - 1

    alpha = mean * tmp
    beta = (1 - mean) * tmp

    return alpha, beta


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
    # sum the one_hot_matrix elements selected for each sample
    ys = np.clip(ys.sum(axis=-2), 0, 1)

    return ys


if __name__ == "__main__":

    SEED = int(time())

    key = jax.random.PRNGKey(SEED)

    N_TOT_FEATURES = 256
    N_SAMPLES = 5000
    N_NOISED = 20
    N_ACTIVE_FEATURES = [1, 2, 3, 4, 5, 6, 7]
    # mean and variances of the beta distribution used to noise the samples
    MEAN_VARS = np.array(
        [
            [0.005, 0.001],
            [0.005, 0.003],
            [0.01, 0.001],
            [0.01, 0.005],
            [0.01, 0.009],
            [0.03, 0.001],
            [0.03, 0.005],
            [0.03, 0.01],
            [0.05, 0.001],
            [0.05, 0.005],
            [0.05, 0.01],
        ]
    )
    BETAS, ALPHAS = get_beta_dist_params(MEAN_VARS[:, 0], MEAN_VARS[:, 1])

    MIs = []

    for n_active in N_ACTIVE_FEATURES:

        print(f"n_active: {n_active}")

        MIs.append([])

        for alpha, beta in zip(ALPHAS, BETAS):

            print(f"\talpha: {alpha}, beta: {beta}")

            t = time()

            # generate random binary samples with exactly n_active active units each
            key, _ = jax.random.split(key)
            ys = gen_ys_k_active(key, n_active, N_SAMPLES, N_TOT_FEATURES)

            # generate a distribution of beta values with N_TOT_FEATURES for each sample
            key, _ = jax.random.split(key)
            p_noise = jax.random.beta(key, alpha, beta, shape=ys.shape)
            # draw the noise from the p_noise as a bernoulli
            # NOTE, for the "positive" samples we generate multiple versions for each sample
            key_positive, key_negative = jax.random.split(key)
            ys_noise_pos = jax.random.bernoulli(
                key_positive, p_noise, shape=(N_NOISED, *ys.shape)
            )
            ys_noise_neg = jax.random.bernoulli(
                key_negative, 1.0 - p_noise, shape=(N_NOISED, *ys.shape)
            )

            # create the noised samples
            ys_noise = np.logical_or(ys, ys_noise_pos)
            ys_noise = np.logical_and(ys_noise, ys_noise_neg)

            print(f"Total time: {time() - t}")
            exit()

            # compute the value of the critic
            # g(x,y) = ln(p_ij[i,j]) with p_ij[i,j] = p(y_i | y_j)
            p_ij = sdm.expected_custom_index(ys[:, None, :], ys[None, :, :])
            # p_ij = sdm.expected_Jaccard_index(ys[:, None, :], ys[None, :, :])
            # p_ij = sdm.expected_and(ys[:, None, :], ys[None, :, :])

            # # evaluate the partition function associated with the tilting function
            # the exponent is removed because the critic g(x,y) has logarithm, they cancel out
            z_i = p_ij.mean(axis=-1)

            pmi = np.log((np.diag(p_ij) / (z_i + 1e-6)) + 1e-6)
            # pmi = np.log(1.0/(z_i + 1e-6))
            # but if ys has zero active units, the pmi is actually 0
            pmi = pmi * (ys.sum(axis=-1) > 1e-6)

            MI = pmi.mean(axis=-1)

            # print(f"n_k: {n_k}")
            # print(f"\t MI: {MI}")

            MIs[-1].append(MI)

    MIs = np.array(MIs)

    # plot the curves of MI vs n_k_bernoulli obtained with different n_k

    fig = go.Figure()

    for i, n_k in enumerate(n_ks):
        fig.add_trace(
            go.Scatter(
                x=n_ks_noise,
                y=MIs[i],
                mode="lines+markers",
                name=f"{n_k}",
                # line properties
                line=dict(width=2),
                # marker properties
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=f"MI vs Added Noise - {n_yi} Features",
        xaxis_title="Noise level (mean extra active units)",
        yaxis_title="MI",
        legend_title="Baseline active features<br>(no noise)",
        # set fig style to white
        plot_bgcolor="white",
        paper_bgcolor="white",
        # and grey gridlines
        xaxis=dict(gridcolor="lightgrey"),
        yaxis=dict(gridcolor="lightgrey"),
        # tick every unit on the x axis
        xaxis_dtick=1,
        # and every 0.5 units on the y axis
        yaxis_dtick=0.5,
    )

    fig.show()
