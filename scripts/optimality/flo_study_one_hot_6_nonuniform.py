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

"""
Version implementing non-uniform sampling of which units are active.
"""
# non uniform

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

    # Calculate Beta distribution parameters based on mean and concentration
    # alpha + beta = concentration
    # alpha / (alpha + beta) = mean_p
    alpha = concentration * mean_p
    beta = concentration * (1.0 - mean_p)

    # Ensure alpha and beta are positive (handled by input validation)
    # JAX's beta function requires positive parameters.

    # Sample probabilities from the Beta distribution
    ps = jax.random.beta(key, alpha, beta, shape=(n_batch_size, n_total_features))

    # Clip probabilities to avoid potential numerical issues at exact 0 or 1
    # although Beta distribution naturally stays within (0, 1) for alpha, beta > 0
    ps = np.clip(ps, np.finfo(ps.dtype).eps, 1.0 - np.finfo(ps.dtype).eps)

    return ps, (alpha, beta)


similarity_dict = {
    "jaccard": partial(sbdr.expected_Jaccard_index, eps=1e-8),
    "gamma": partial(sbdr.gamma_similarity, eps=1e-8),
    "and": sbdr.expected_and,
    "proxy_jaccard": partial(sbdr.proxy_jaccard_index, eps=1e-8),
    "asymmetric_jaccard": partial(sbdr.asymmetric_jaccard_similarity, eps=1e-8),
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

    SIMILARITY = "proxy_jaccard"
    N_TOT_FEATURES = 256
    N_SINGLE_MASK_SAMPLES = 100
    N_MASK_P_RESAMPLES = 200
    N_MASK_MEAN_ACTIVE = [1.0, 2.0, 3.0, 4.0]
    MASK_CONCENTRATIONS = [0.2, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    MASK_MULTIPLIERS = [0.1, 0.2, 0.5, 0.9, 1.0]
    # NOTE visualize Beta distribution PDF
    # here: https://www.acsu.buffalo.edu/~adamcunn/probability/beta.html
    # MEAN_NOISE_UNITS = [0, 1, 2, 3, 4, 5, 6]
    # NOISE_NEG = 0.2

    MIs = []

    verbose = False
    disable_tqdm = verbose

    for n_mask_mean_active in tqdm(N_MASK_MEAN_ACTIVE, disable=disable_tqdm):

        MIs.append([])

        if verbose:
            print(f"\nn_mask_mean_active: {n_mask_mean_active}")

        for i, mask_concentration in enumerate(
            tqdm(MASK_CONCENTRATIONS, leave=False, disable=disable_tqdm)
        ):

            MIs[-1].append([])

            if verbose:
                print(f"  mask_concentration: {mask_concentration}")

            for mask_multiplier in tqdm(
                MASK_MULTIPLIERS, leave=False, disable=disable_tqdm
            ):

                if verbose:
                    print(f"\tmask_multiplier: {mask_multiplier}")

                # Generate the probabilities for the activation mask
                key, _ = jax.random.split(key)

                mask_ps, (alpha, beta) = generate_ps_beta_distribution(
                    key,
                    n_mean_active=n_mask_mean_active,
                    n_total_features=N_TOT_FEATURES,
                    concentration=mask_concentration,
                    n_batch_size=N_MASK_P_RESAMPLES,
                )

                if verbose:
                    print(f"\t  alpha: {alpha}, beta: {beta}")

                # Generate the N_SAMPLES samples from the activation mask
                key, _ = jax.random.split(key)
                zs_mask = jax.random.bernoulli(
                    key, mask_ps, shape=(N_SINGLE_MASK_SAMPLES, *(mask_ps.shape))
                )
                zs_mask = zs_mask.reshape((-1, N_TOT_FEATURES))

                # print(zs_mask.shape)
                # print(zs_mask.sum(axis=-1).mean(axis=-1))
                # exit()

                # Apply the mask
                zs = mask_multiplier * zs_mask
                # print(zs.mean(), zs.std(), zs.min(), zs.max())
                # zs = np.clip(zs, 0.001, 1)

                sim_fn = similarity_dict[SIMILARITY]
                # self-similarity
                p_ii = sim_fn(zs, zs)
                # cross-similarity
                p_ij = sim_fn(zs[:, None, ...], zs[None, :, ...])

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
    print(f"MIs.shape: {MIs.shape}")
    # exit()
    # # # Plot the curves of MI

    fig = go.Figure()

    for i, n_mask_mean_active in enumerate(N_MASK_MEAN_ACTIVE):
        for j, mask_multiplier in enumerate(MASK_MULTIPLIERS):

            hue = i / len(N_MASK_MEAN_ACTIVE)
            lightness = 0.5 + mask_multiplier * 0.3
            saturation = 0.4 + mask_multiplier * 0.3
            color = generate_color(hue, lightness, saturation)

            fig.add_trace(
                go.Scatter(
                    # x=np.arange(len(MASK_CONCENTRATIONS)),
                    y=MIs[i, :, j],
                    mode="lines+markers",
                    name=f"{n_mask_mean_active} n_feat, {mask_multiplier:.2f} p_mult",
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
                    legendgroup=f"{n_mask_mean_active}",
                    # # On hovering, show the n_mask_mean_active and the p_baseline
                    hovertemplate=(
                        f"mean_active: {n_mask_mean_active}<br>p_multiplier: {mask_multiplier:.3f}"
                    ),
                )
            )

    fig.update_layout(
        title=f"MI non-uniform mask - {N_TOT_FEATURES} Features<br>Similarity: {SIMILARITY}<br>Mask resamples: {N_MASK_P_RESAMPLES}<br>Samples per single mask: {N_SINGLE_MASK_SAMPLES}",
        xaxis_title="Concentration",
        yaxis_title="MI",
        # legend_title="Baseline active features<br>(no noise)",
        # set fig style to white
        plot_bgcolor="white",
        paper_bgcolor="white",
        # and grey gridlines
        xaxis=dict(gridcolor="lightgrey"),
        yaxis=dict(gridcolor="lightgrey", range=[MIs.min() - 1, MIs.max() + 1]),
        # tick every unit on the x axis
        xaxis_dtick=1,
        # and every 0.5 units on the y axis
        yaxis_dtick=0.5,
    )

    print("AAAAAAAAAAAAAAAAAAAAAAA")

    fig.show()

    if SAVE_FIGS:
        fig_name = f"mi_non_uniform_mask_{SIMILARITY}_{N_TOT_FEATURES}_mask-resamples-{N_MASK_P_RESAMPLES}_single-mask-samples-{N_SINGLE_MASK_SAMPLES}"
        fig_path = os.path.join(fig_base_path, fig_name + ".html")
        fig.write_html(fig_path, include_plotlyjs="directory")
