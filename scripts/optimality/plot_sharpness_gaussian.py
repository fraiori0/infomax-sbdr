import os
import jax
import jax.numpy as np
from jax import grad, jit, vmap
import infomax_sbdr as sbdr
from time import time
from tqdm import tqdm
from functools import partial
from sampling_fn import *
import json
import plotly.graph_objects as go

# Online tool to visualize the pdf of the Beta distribution
# https://www.acsu.buffalo.edu/~adamcunn/probability/beta.html

if __name__ == "__main__":

    SAVE_NAME = "sharpness_f128_s14400_multi"

    data_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "sampled_mi",
        "sharpness_gaussian",
    )

    SAVE = True

    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "figs",
        "infomax optimality",
        "sharpness_gaussian",
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # import data
    with open(os.path.join(data_folder, SAVE_NAME + ".json"), "r") as f:
        data = json.load(f)

    # MI shape - {sim_name: (n_nonzero, noise_p, seeds)}
    MIs = data["MI"]
    for k in MIs.keys():
        MIs[k] = np.array(MIs[k])
    n_nonzero = data["n_nonzero"]
    gamma_concentration = data["gamma_concentration"]
    mu = np.array(data["mu"])
    sigma = np.array(data["sigma"])
    seeds = data["seeds"]

    # plot

    fig = go.Figure()

    for sim_name in MIs.keys():
        for mi, n_nz in zip(MIs[sim_name], n_nonzero):
            hue = n_nz / max(7, max(n_nonzero))
            lightness = 0.5
            saturation = 0.7
            color = generate_color(hue, lightness, saturation)

            # compute std for each gamma_concentration
            mi_mean = mi.mean(axis=-1)
            mi_std = mi.std(axis=-1)

            # plot with error bars
            fig.add_trace(
                go.Scatter(
                    x=mu,
                    y=mi_mean,
                    error_y=dict(
                        type="data",
                        array=mi_std,
                    ),
                    mode="lines+markers",
                    marker=dict(
                        color=color,
                        size=10,
                    ),
                    line=dict(
                        width=2,
                        color=color,
                    ),
                    name=f"{sim_name} {n_nz} features",
                    legendgroup=f"{sim_name}",
                )
            )

    fig.update_layout(
        xaxis_title="Gamma concentration",
        yaxis_title="Mutual Information (bits)",
        width=800,
        height=600,
        legend_title="Average Non-zero Features",
        font=dict(size=14),
        template="plotly_white",
    )

    # set y axis properties
    fig.update_yaxes(
        range=[0.0, 6.0],  # MIs.max() + 0.5], #
        # ticks every 0.5
        dtick=0.5,
    )

    fig.show()

    if SAVE:
        fig.write_html(
            os.path.join(save_folder, SAVE_NAME + ".html"),
            include_plotlyjs="directory",
        )
