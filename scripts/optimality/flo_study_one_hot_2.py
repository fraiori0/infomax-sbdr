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
import colorsys


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


def generate_color(h, l, s):
    """Convert HLS values (range [0,1]) to an rgb string for Plotly."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


if __name__ == "__main__":

    SEED = int(time())

    key = jax.random.PRNGKey(SEED)

    N_TOT_FEATURES = 256
    N_SAMPLES = 2000
    N_NONZERO_FEATURES = [1, 2, 3, 5, 7]  #
    # combinations of mean and variance of the normal distribution
    # to be tested
    MUS = [1.0]
    SIGMAS = [3.0, 1.0, 0.1]
    MEAN_NOISE_UNITS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    MIs = []

    disable_tqdm = False
    verbose = False

    for n_nonzero_features in tqdm(N_NONZERO_FEATURES, disable=disable_tqdm):

        MIs.append([])

        if verbose:
            print(f"\nnonzero features: {n_nonzero_features}")

        for mu in tqdm(MUS, leave=False, disable=disable_tqdm):

            MIs[-1].append([])

            if verbose:
                print(f"  mu: {mu}")

            for sigma in tqdm(SIGMAS, leave=False, disable=disable_tqdm):

                MIs[-1][-1].append([])

                if verbose:
                    print(f"\tsigma: {sigma}")

                for mean_noise_units in tqdm(
                    MEAN_NOISE_UNITS, leave=False, disable=disable_tqdm
                ):

                    if verbose:
                        print(f"\t  noise units: {mean_noise_units}")

                    # Generate samples
                    key, _ = jax.random.split(key)
                    zs = zs = jax.random.normal(key, shape=(N_SAMPLES, N_TOT_FEATURES))
                    zs = zs * sigma + mu
                    # clip in [0, 1]
                    zs = np.clip(zs, 1e-6, 1.0)

                    # Generate mask with fixed amount of non-zero features
                    key, _ = jax.random.split(key)
                    zs_mask_fixed = gen_ys_k_active(
                        key, n_nonzero_features, N_SAMPLES, N_TOT_FEATURES
                    )

                    # Generate masks from a bernoulli with an average of mean_noise_units
                    key, key_pos, key_neg = jax.random.split(key, 3)
                    # Mask with a few True value, this units will be included
                    zs_mask_noise_pos = jax.random.bernoulli(
                        key_pos,
                        p=mean_noise_units / N_TOT_FEATURES,
                        shape=(N_SAMPLES, N_TOT_FEATURES),
                    )
                    # Mask with a few False value, this units will be excluded
                    zs_mask_noise_neg = jax.random.bernoulli(
                        key_neg,
                        p=(1.0 - mean_noise_units / N_TOT_FEATURES),
                        shape=(N_SAMPLES, N_TOT_FEATURES),
                    )

                    # Combine the masks
                    zs_mask = np.logical_or(zs_mask_fixed, zs_mask_noise_pos)
                    zs_mask = np.logical_and(zs_mask, zs_mask_noise_neg)

                    # Apply the mask
                    zs = zs * zs_mask

                    # print(zs.sum(axis=-1))
                    # print(zs.sum(axis=-1).mean())
                    # print(zs.sum(axis=-1).std())
                    # exit()

                    sim_fn = sbdr.gamma_similarity
                    p_ii = sim_fn(zs, zs)
                    p_ij = sim_fn(zs[:, None, :], zs[None, :, :])

                    # check for nans in p_ij
                    # check if any sample has all zeros
                    # if np.any(np.isnan(p_ij)):
                    #     print(
                    #         f"\nf: {n_nonzero_features} - mu: {mu} - sigma: {sigma} - nf: {mean_noise_units}"
                    #     )
                    #     print(p_ii.min())

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
                    if np.any(np.isnan(pmis)):
                        raise ValueError("pmis has nan")

                    mi = pmis.mean(axis=-1)

                    MIs[-1][-1][-1].append(mi)

    MIs = np.array(MIs)

    print(MIs.shape)

    # # # Plot the curves of MI

    fig = go.Figure()

    max_y = 0

    for i, nonzero_feature in enumerate(N_NONZERO_FEATURES):
        for j, mu in enumerate(MUS):
            for k, sigma in enumerate(SIGMAS):
                hue = i / len(N_NONZERO_FEATURES)
                lightness = 0.4 + (k / len(SIGMAS)) * 0.4
                saturation = 0.8  # 0.4 + (j/len(MUS)) * 0.4
                color = generate_color(hue, lightness, saturation)
                # print(MIs[i, j, k])
                # exit()
                fig.add_trace(
                    go.Scatter(
                        x=MEAN_NOISE_UNITS,
                        y=MIs[i, j, k],
                        mode="lines+markers",
                        name=f"{nonzero_feature} n_feat, {mu} mu, {sigma} sigma",
                        # line properties
                        line=dict(
                            width=2,
                            color=color,
                        ),
                        # marker properties
                        marker=dict(
                            size=6,
                            color=color,
                        ),
                        legendgroup=f"{mu} mu {sigma} sigma",
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
