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

    SAVE_NAME = "concentration_f256_s10000"

    data_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "sampled_mi",
        "concentration",
    )

    SAVE = True

    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "figs",
        "infomax optimality",
        "concentration",
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # import data
    with open(os.path.join(data_folder, SAVE_NAME + ".json"), "r") as f:
        data = json.load(f)

    
    MIs = data["MI"]
    for k in MIs.keys():
        # MI shape - (n_nonzero, gamma_concentration, seeds)
        MIs[k] = np.array(MIs[k])
    n_nonzero = data["n_nonzero"]
    gamma_concentration = data["gamma_concentration"]
    uniform_range = data["uniform_range"]
    seeds = data["seeds"]

    # plot

    fig = go.Figure()

    marker_symbols = ["circle", "x", "triangle-up", "square", "diamond", "cross"]
    line_styles = ["solid", "dash", "dot", "dashdot", "longdash"]

    for k, sim_k in enumerate(MIs.keys()):
        for i, conc in enumerate(gamma_concentration):
            # select a single concentration
            mi = MIs[sim_k][:, i, :]  # shape (n_nonzero, seeds)
    
            hue = i / len(gamma_concentration)
            lightness = 0.5
            saturation = 0.7
            color = generate_color(hue, lightness, saturation)

            # compute std for each gamma_concentration
            mi_mean = mi.mean(axis=-1)
            mi_std = mi.std(axis=-1)

            # plot with error bars
            fig.add_trace(
                go.Scatter(
                    x=n_nonzero,
                    y=mi_mean,
                    error_y=dict(
                        type="data",
                        array=mi_std,
                    ),
                    mode="lines+markers",
                    name=f"{conc:.3f} - {sim_k}",
                    marker=dict(
                        color=color,
                        size=10,
                        symbol=marker_symbols[k],
                    ),
                    line=dict(
                        width=2,
                        color=color,
                        dash=line_styles[k],
                    ),
                    legendgroup=f"{sim_k}"
                )
            )

    fig.update_layout(
        xaxis_title="Average non-zero units",
        yaxis_title="Mutual Information (bits)",
        width=800,
        height=600,
        legend_title="Average Non-zero Features",
        font=dict(size=14),
        template="plotly_white",
    )

    # set x axis to log scale
    # fig.update_xaxes(type="log")
    # set y axis properties
    fig.update_yaxes(
        # range=[0.0, 6.0],  # MIs.max() + 0.5], #
        # ticks every 0.5
        dtick=0.5,
    )

    fig.show()

    if SAVE:
        fig.write_html(
            os.path.join(save_folder, SAVE_NAME + ".html"),
            include_plotlyjs="directory",
        )
