import os
import sys
import jax
import jax.numpy as np
from jax import grad, jit, vmap
import infomax_sbdr as sbdr
import plotly.graph_objects as go
from jax.experimental import sparse
from time import time
import argparse
from tqdm import tqdm
import colorsys
from functools import partial
from utils import *

# Online tool to visualize the pdf of the Beta distribution
# https://www.acsu.buffalo.edu/~adamcunn/probability/beta.html

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

    SEED = 1  # int(time())

    key = jax.random.key(SEED)

    SIMILARITY = "jaccard"

    # Number of features (i.e., dimension of a single sample)
    N_FEATURES = 128
    # Number of samples drawn from the gamma distribution
    N_GAMMA_SAMPLES = 100
    # Number of binary samples drawn from each gamma distribution
    N_SINGLE_MASK_SAMPLES = 200
    # Expected number of non-zero units after Bernoulli-sampling from N_FEATURES gamma-distributed probabilities over
    GAMMA_MU = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Concentration of the gamma distribution
    GAMMA_CONCENTRATIONS = [0.1, 0.2, 0.5, 1.0, 10.0, 100.0]
    # Range for the activation level, after selecting the non-zero units
    UNIFORM_RANGE = [
        (0.1, 1.0),
        (0.2, 1.0),
        (0.5, 1.0),
        (0.8, 1.0),
        (0.9, 1.0),
    ]

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

            for activation_range in tqdm(
                ACTIVATION_RANGES, leave=False, disable=disable_tqdm
            ):

                if verbose:
                    print(f"\tactivation_range: {activation_range}")

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

                # # Sample uniformly from the activation_range the activation value of each unit
                key, _ = jax.random.split(key)
                activation_values = jax.random.uniform(
                    key,
                    shape=zs_mask.shape,
                    minval=activation_range[0],
                    maxval=activation_range[1],
                )

                # Apply the mask
                zs = activation_values * zs_mask
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
        for j, activation_range in enumerate(ACTIVATION_RANGES):

            act_min = activation_range[0]
            act_max = activation_range[1]

            hue = i / len(N_MASK_MEAN_ACTIVE)
            lightness = 0.5 + act_min * 0.3
            saturation = 0.4 + act_max * 0.3
            color = generate_color(hue, lightness, saturation)

            fig.add_trace(
                go.Scatter(
                    x=np.log(np.array(MASK_CONCENTRATIONS) + 1),
                    y=MIs[i, :, j],
                    mode="lines+markers",
                    name=f"{n_mask_mean_active} n_feat, {act_min:.2f}-{act_max:.2f} act_range",
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
                        f"mean_active: {n_mask_mean_active}<br>activation range_multiplier: {act_min:.2f}-{act_max:.2f}"
                    ),
                )
            )

    fig.update_layout(
        title=f"MI non-uniform mask non-uniform activation - {N_TOT_FEATURES} Features<br>Similarity: {SIMILARITY}<br>Mask resamples: {N_MASK_P_RESAMPLES}<br>Samples per single mask: {N_SINGLE_MASK_SAMPLES}",
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

    fig.show()

    if SAVE_FIGS:
        fig_name = f"mi_non-uniform_mask-act_{SIMILARITY}_{N_TOT_FEATURES}_mask-resamples-{N_MASK_P_RESAMPLES}_single-mask-samples-{N_SINGLE_MASK_SAMPLES}"
        fig_path = os.path.join(fig_base_path, fig_name + ".html")
        fig.write_html(fig_path, include_plotlyjs="directory")
