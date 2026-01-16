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
import plotly.io as pio

# ε

# Online tool to visualize the pdf of the Beta distribution
# https://www.acsu.buffalo.edu/~adamcunn/probability/beta.html

if __name__ == "__main__":

    # set plotly render output to browser
    pio.renderers.default = "browser"

    SAVE_NAME = "sharpness_f256_s10000"

    sim_names = {
        "log_and": r"$g_{\epsilon}$",
    }
    fig_size={
        "width": 800,
        "height": 500
    }

    font_family = "Latin Modern Roman"

    data_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "sampled_mi",
        "sharpness",
    )

    SAVE = True

    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "results",
        "infomax optimality",
        "sharpness",
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # import data
    with open(os.path.join(data_folder, SAVE_NAME + ".json"), "r") as f:
        data = json.load(f)

    MIs = data["MI"]
    for k in MIs.keys():
        # MI shape - (n_nonzero, gamma_concentration, seeds, eps_sim)
        MIs[k] = np.array(MIs[k])
    n_nonzero = data["n_nonzero"]
    gamma_concentration = data["gamma_concentration"]
    uniform_range = np.array(data["uniform_range"])
    seeds = data["seeds"]
    eps_sim = np.array(data["eps_sim"])

    # plot

    fig = go.Figure()

    marker_symbols = ["circle", "x", "triangle-up", "square", "diamond", "cross"]
    line_styles = ["solid", "dash", "dot", "dashdot", "longdash"]

    for k, sim_k in enumerate(sim_names):
        for i, u_range in enumerate(uniform_range):
            for j, e_sim in enumerate(eps_sim):
                # select a single concentration
                mi = MIs[sim_k][:, i, :, j]  # shape (n_nonzero, seeds)
        
                hue = i / len(uniform_range)
                lightness = 0.6
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
                        name=str(round(float(u_range[0]), 3)),
                        marker=dict(
                            color=color,
                            size=10,
                            symbol=marker_symbols[j],
                        ),
                        line=dict(
                            width=2,
                            color=color,
                            dash=line_styles[j],
                        ),
                        legend=f"legend{j+1}",
                    )
                )

    fig.update_xaxes(
        title=dict(
            text=r"Average number of non-zero units",
            font=dict(size=20, family=font_family)),
        tickfont=dict(size=18, family=font_family),
    )

    fig.update_yaxes(
        title=dict(
            text=r"Mutual Information (nats)",
            font=dict(size=20, family=font_family)),
        tickfont=dict(size=18, family=font_family),
    )

    fig.update_layout(
        **fig_size,
        # font=dict(size=20, family=font_family),
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

    # Position the legends
    # position the legend for this subplot
    for i, e_sim in enumerate(eps_sim):
        fig.update_layout(
            **{f"legend{i+1}":
               dict(
                   x=1.1,
                   y=1.0 - 0.6 * i,
                   xanchor='left',
                   yanchor='top',
                   title=r"$a_{min} \; (\epsilon = " + str(e_sim) + ")$",
                   font=dict(size=18, family=font_family),
                    # title font size
                    title_font=dict(size=20, family=font_family),
            ),}
        )
        # # also set font for all legends
        # fig.layout.annotations[i].font = dict(size=18, family="Times New Roman")

    fig.show()

    if SAVE:
        # export as svg
        fig.write_image(
            os.path.join(save_folder, SAVE_NAME + ".svg"),
            format="svg",
            **fig_size,
            scale=3,
        )