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

    SIM_KEY = "logand"

    SAVE_NAME = "simbias_f256_s10000"

    data_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "sampled_mi",
        "simbias",
    )

    SAVE = True

    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "figs",
        "infomax optimality",
        "simbias",
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # import data
    with open(os.path.join(data_folder, SAVE_NAME + ".json"), "r") as f:
        data = json.load(f)

    MIs = data["MI"]
    for k in MIs.keys():
        MIs[k] = np.array(MIs[k])
    
    n_nonzero = data["n_nonzero"]
    gamma_concentration = data["gamma_concentration"]
    uniform_range = data["uniform_range"]
    eps_cross = np.array(data["eps_cross"])
    eps_self = np.array(data["eps_self"])
    seeds = data["seeds"]

    # plot

    fig = go.Figure()

    f# select a single key
    MIs = MIs[SIM_KEY]  # shape (n_nonzero, seeds, eps_self, eps_cross)
        
    for i, n_nz in enumerate(n_nonzero):
        for j, e_slf in enumerate(eps_self):
            mi = MIs[i, :, :, j]  # mi : shape (eps_cross, seeds)
    
            hue = i / len(n_nonzero)
            lightness = 0.5 + 0.25 * (j / len(eps_self))
            saturation = 0.7
            color = generate_color(hue, lightness, saturation)

            # compute mean and std over the seeds
            mi_mean = mi.mean(axis=-1)
            mi_std = mi.std(axis=-1)

            # plot with error bars
            fig.add_trace(
                go.Scatter(
                    x=eps_cross,
                    y=mi_mean,
                    error_y=dict(
                        type="data",
                        array=mi_std,
                    ),
                    mode="lines+markers",
                    name=f"{n_nz} F - e {e_slf:.3f}",
                    marker=dict(
                        color=color,
                        size=10,
                    ),
                    line=dict(
                        width=2,
                        color=color,
                    ),
                    legendgroup=f"{j} eps_self",
                )
            )

    fig.update_layout(
        xaxis_title="Epsilon Cross",
        yaxis_title="Mutual Information (nats)",
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
