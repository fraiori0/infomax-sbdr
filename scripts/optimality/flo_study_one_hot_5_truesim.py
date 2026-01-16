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
from functools import partial


# # Create an argument parser
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument(
#     "-s",
#     "--seed",
#     type=str,
#     help="Path to the folder containing the files for the validation set",
#     default=None,
# )
# parser.add_argument(
#     "-n", "--name", type=str, help="Name of this Optuna study", default="test"
# )


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


def get_beta_dist_mean_std(alpha, beta):
    """
    Get the mean and std parameters for a Beta distribution with the given alpha and beta parameters.

    Note, this function is compatible with broadcastable mean and variance arrays.
    """

    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    std = np.sqrt(variance)

    return mean, std


def generate_color(h, l, s):
    """Convert HLS values (range [0,1]) to an rgb string for Plotly."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


similarity_dict = {
    "jaccard": partial(sbdr.expected_Jaccard_index, eps=1e-8),
    "gamma": partial(sbdr.gamma_similarity, eps=1e-8),
    "and": sbdr.expected_and,
}


if __name__ == "__main__":

    SAVE_FIGS = True
    fig_base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "figs",
        "infomax optimality",
    )
    if not os.path.exists(fig_base_path):
        os.makedirs(fig_base_path)

    SEED = int(time())

    key = jax.random.PRNGKey(SEED)

    SIMILARITY = "jaccard"
    N_TOT_FEATURES = 128
    N_SAMPLES = 128
    N_RESAMPLES = 30
    N_MI_SAMPLES = 30
    N_NONZERO_FEATURES = [0, 1, 2, 3, 4]  #
    # combinations of mean and variance of the normal distribution
    # to be tested
    # note: check values from here https://www.acsu.buffalo.edu/~adamcunn/probability/beta.html
    ALPHAS_BETAS = np.array(
        (
            (90, 1.5),
            # (115, 13),
            (25, 2.5),
            # (100, 25),
            (9, 4),
            # (130, 60),
            # (7, 7),
            (1.5, 1.6),
            # (2.5, 0.75),
            (7.0, 14.0),
            (0.001, 0.001),
        ),
    )
    MEAN_BETA, STD_BETA = get_beta_dist_mean_std(ALPHAS_BETAS[:, 0], ALPHAS_BETAS[:, 1])

    MEAN_NOISE_UNITS = [0, 1, 2, 3, 4, 5, 6]
    NOISE_NEG = 0.2

    MIs = []

    disable_tqdm = False
    verbose = False

    for n_nonzero_features in tqdm(N_NONZERO_FEATURES, disable=disable_tqdm):

        MIs.append([])

        if verbose:
            print(f"\nnonzero features: {n_nonzero_features}")

        for i, beta_params in enumerate(
            tqdm(ALPHAS_BETAS, leave=False, disable=disable_tqdm)
        ):

            MIs[-1].append([])

            alpha, beta = beta_params[0], beta_params[1]

            if verbose:
                print(f"  beta: {beta}, alpha: {alpha}")

            for mean_noise_units in tqdm(
                MEAN_NOISE_UNITS, leave=False, disable=disable_tqdm
            ):

                if verbose:
                    print(f"\tnoise units: {mean_noise_units}")

                # # Generate samples
                # note, last one is substituted for an array with sharp 1 values
                if i < (len(beta_params) - 1):
                    key, _ = jax.random.split(key)
                    zs = jax.random.beta(
                        key,
                        alpha,
                        beta,
                        shape=(N_MI_SAMPLES, N_SAMPLES, N_TOT_FEATURES),
                    )
                else:
                    zs = np.ones((N_MI_SAMPLES, N_SAMPLES, N_TOT_FEATURES))

                # Generate mask with fixed amount of non-zero features
                if n_nonzero_features == 0:
                    zs_mask_fixed = np.zeros(
                        (N_MI_SAMPLES, N_SAMPLES, N_TOT_FEATURES), dtype=bool
                    )
                else:
                    key, _ = jax.random.split(key)
                    zs_mask_fixed = gen_ys_k_active(
                        key,
                        n_nonzero_features,
                        N_MI_SAMPLES * N_SAMPLES,
                        N_TOT_FEATURES,
                    )
                    zs_mask_fixed = zs_mask_fixed.reshape(
                        (N_MI_SAMPLES, N_SAMPLES, N_TOT_FEATURES)
                    )

                # Generate masks from a bernoulli with an average of mean_noise_units
                key, key_pos, key_neg = jax.random.split(key, 3)
                # Mask with a few True value, this units will be included
                zs_mask_noise_pos = jax.random.bernoulli(
                    key_pos,
                    p=float(mean_noise_units) / N_TOT_FEATURES,
                    shape=(N_MI_SAMPLES, N_SAMPLES, N_TOT_FEATURES),
                )
                # Mask with a few False value, this units will be excluded
                zs_mask_noise_neg = jax.random.bernoulli(
                    key_neg,
                    p=1.0 - NOISE_NEG,
                    shape=(N_MI_SAMPLES, N_SAMPLES, N_TOT_FEATURES),
                )

                # Combine the masks
                zs_mask = np.logical_or(zs_mask_fixed, zs_mask_noise_pos)
                zs_mask = np.logical_and(zs_mask, zs_mask_noise_neg)

                # Apply the mask
                zs = zs * zs_mask
                # print(zs.mean(), zs.std(), zs.min(), zs.max())
                zs = np.clip(zs, 0.001, 1)

                # # # Sample from zs like it was a multivariate Bernoulli
                # each sample distribution is used to create a batch of N_RESAMPLES
                key, _ = jax.random.split(key)
                zs_sampled = jax.random.bernoulli(
                    key,
                    p=zs[..., None, :],
                    shape=(N_MI_SAMPLES, N_SAMPLES, N_RESAMPLES, N_TOT_FEATURES),
                )

                sim_fn = similarity_dict[SIMILARITY]
                # self-similarity is the mean between the N_RESAMPLES from the same distribution
                p_ii = sim_fn(
                    zs_sampled[..., :, None, :], zs_sampled[..., None, :, :]
                ).mean(axis=(-2, -1))
                # cross-similarity is computed between the N_RESAMPLES from different distributions
                # to avoi docmbinatorial explosion of computations
                p_ij = sim_fn(
                    zs_sampled[:, :, None, ...], zs_sampled[:, None, :, ...]
                ).mean(axis=-1)

                # check for nans in p_ij

                if np.any(np.isnan(p_ij)):
                    raise ValueError("p_ij has nan")
                # print(p_ij.min())
                # print(p_ij.max())

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
                pmis = np.log(p_ii / (p_ij.mean(axis=-2) + 1e-6) + 1e-6)
                if np.any(np.isnan(pmis)):
                    raise ValueError("pmis has nan")

                mi = pmis.mean(axis=-1)

                MIs[-1][-1].append(mi)

    MIs = np.array(MIs)

    # # # Plot the curves of MI

    fig = go.Figure()

    for i, nonzero_feature in enumerate(N_NONZERO_FEATURES):
        for j, _ in enumerate(ALPHAS_BETAS):
            mean = MEAN_BETA[j]
            std = STD_BETA[j]
            alpha = ALPHAS_BETAS[j, 0]
            beta = ALPHAS_BETAS[j, 1]
            # normalize the mean and std in 0-1
            mean_norm = (mean - MEAN_BETA.min()) / (MEAN_BETA.max() - MEAN_BETA.min())
            std_norm = (std - STD_BETA.min()) / (STD_BETA.max() - STD_BETA.min())

            hue = i / len(N_NONZERO_FEATURES)
            lightness = 0.5 + mean_norm * 0.3
            saturation = 0.4 + std_norm * 0.3
            color = generate_color(hue, lightness, saturation)

            mi_means = np.mean(MIs[i, j], axis=-1)
            mi_bounds = np.quantile(MIs[i, j], np.array([0.1, 0.9]), axis=-1)

            fig.add_trace(
                go.Scatter(
                    x=MEAN_NOISE_UNITS,
                    y=mi_means,
                    mode="lines+markers",
                    name=f"{nonzero_feature} n_feat, {mean:.3f} mu, {std:.4f} std - Mean",
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
                    legendgroup=f"{nonzero_feature} - {mean}",  # f"{mean}",  #
                    # On hovering, show the mean, the std, the alpha and the beta parameters
                    hovertemplate=(
                        f"{mean:.3f} mu, {std:.4f} std<br>{alpha:.3f} alpha, {beta:.3f} beta"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=MEAN_NOISE_UNITS,
                    y=mi_bounds[0],
                    mode="lines",
                    name=f"{nonzero_feature} n_feat, {mean:.3f} mu, {std:.4f} std",
                    # line properties
                    line=dict(
                        width=1,
                        color=color,
                        dash="dot",
                    ),
                    legendgroup=f"{nonzero_feature} - {mean} - percentile",  # f"{mean}",  #
                    # On hovering, show the mean, the std, the alpha and the beta parameters
                    hovertemplate=(
                        f"{mean:.3f} mu, {std:.4f} std<br>{alpha:.3f} alpha, {beta:.3f} beta"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=MEAN_NOISE_UNITS,
                    y=mi_bounds[1],
                    mode="lines",
                    name=f"{nonzero_feature} n_feat, {mean:.3f} mu, {std:.4f} std",
                    # line properties
                    line=dict(
                        width=1,
                        color=color,
                        dash="dash",
                    ),
                    legendgroup=f"{nonzero_feature} - {mean} - percentile",  # f"{mean}",  #
                    # On hovering, show the mean, the std, the alpha and the beta parameters
                    hovertemplate=(
                        f"{mean:.3f} mu, {std:.4f} std<br>{alpha:.3f} alpha, {beta:.3f} beta"
                    ),
                )
            )

    fig.update_layout(
        title=f"MI resampled quantiles - {N_TOT_FEATURES} Features<br>Similarity: {SIMILARITY}<br>N Samples: {N_SAMPLES}<br>Negative Noise: {NOISE_NEG}",
        xaxis_title="Noise level (mean)",
        yaxis_title="MI",
        # legend_title="Baseline active features<br>(no noise)",
        # set fig style to white
        plot_bgcolor="white",
        paper_bgcolor="white",
        # and grey gridlines
        xaxis=dict(gridcolor="lightgrey"),
        yaxis=dict(gridcolor="lightgrey", range=[0, MIs.max() + 1]),
        # tick every unit on the x axis
        xaxis_dtick=1,
        # and every 0.5 units on the y axis
        yaxis_dtick=0.5,
    )

    print("AAAAAAAAAAAAAAAAAAAAAAA")

    fig.show()

    if SAVE_FIGS:
        fig_name = f"mi_resampled_quantile_{SIMILARITY}_{N_TOT_FEATURES}_features_{N_SAMPLES}_samples_{NOISE_NEG}_noise_neg"
        fig_path = os.path.join(fig_base_path, fig_name + ".html")
        fig.write_html(fig_path, include_plotlyjs="directory")
