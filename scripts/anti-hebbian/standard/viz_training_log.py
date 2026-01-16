import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "3"

import jax
import jax.numpy as np
import numpy as onp
from tbparse import SummaryReader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

np.set_printoptions(precision=4, suppress=True)
pio.renderers.default = "browser"

models = {
    "standard": {
        "3": {"name": r"$p^* = 0.01$", "chkp": 20, "color": "#1f77b4", "dash": "solid", "symbol": "circle", "fillcolor": "rgba(0.12, 0.47, 0.71, 0.2)",},
        "2": {"name": r"$p^* = 0.02$", "chkp": 20, "color": "#be44ff", "dash": "solid", "symbol": "circle", "fillcolor": "rgba(0.75, 0.27, 1.0, 0.2)",},
        "1": {"name": r"$p^* = 0.05$", "chkp": 20, "color": "#2ca02c", "dash": "solid", "symbol": "circle", "fillcolor": "rgba(0.17, 0.63, 0.17, 0.2)",},
        "4": {"name": r"$p^* = 0.075$", "chkp": 20, "color": "#d62728", "dash": "solid", "symbol": "circle", "fillcolor": "rgba(0.84, 0.15, 0.16, 0.2)"},
    },
    # "xor": {
    #     "1": {"name": r"$p_* = 0.075$", "chkp": 30, "color": "#d62728", "dash": "dash", "symbol": "x"},
    # },
}

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    os.pardir,
)
base_folder = os.path.normpath(base_folder)


result_folder = os.path.join(
    base_folder,
    "resources",
    "results",
    "antihebbian",
)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


fig_layout_double_subplot = {
    "height": 500,
    "width": 1000,
}
fig_layout_single = {
    "height": 500,
    "width": 700,   
}

"""---------------------"""
""" Import data """
"""---------------------"""

dfs = {
    k : {} for k in models.keys()
}
for k in models.keys():
    for kk in models[k].keys():
        print(f"{k} - {kk}")

        log_path = os.path.join(
            base_folder,
            "resources",
            "models",
            "antihebbian",
            k,
            kk,
            "logs",
        )
        
        # keep only scalar values and convert to dataframe
        dfs[k][kk] = SummaryReader(log_path).scalars

        # # print all unique value from the "tag" column
        # print(dfs[k][kk]["tag"].drop_duplicates())
        # break


"""---------------------"""
""" Take data for plotting """
"""---------------------"""

key_map = {
    "u25": "unit/0.25/train/epoch",
    "u50": "unit/0.5/train/epoch",
    "u75": "unit/0.75/train/epoch",
    "s25": "sample/0.25/train/epoch",
    "s50": "sample/0.5/train/epoch",
    "s75": "sample/0.75/train/epoch",
}

plot_data = {
    k: {kk: {} for kk in models[k].keys()} for k in models.keys()
}

# for each model, take the 25 and 75 percentile of average unit activation and sample activation, along with the mean
TAKE_FIRST = 20
for k in models.keys():
    for kk in models[k].keys():
        df = dfs[k][kk]
        if TAKE_FIRST is not None:
            plot_data[k][kk] = {
                kkk : (df[df['tag'] == vvv]['value'][:TAKE_FIRST]).tolist()
                for kkk, vvv in key_map.items()
            }
        else:
            plot_data[k][kk] = {
                kkk : (df[df['tag'] == vvv]['value']).tolist()
                for kkk, vvv in key_map.items()
            }



"""---------------------"""
""" Plot training curves """
"""---------------------"""

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Units",
        "Samples",
    ),
    horizontal_spacing=0.15,
    # vertical_spacing=0.15,
)

for k in plot_data.keys():
    for kk in plot_data[k].keys():

        # # # UNITS
        # Unit median
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(plot_data[k][kk]["u50"])),
                y = plot_data[k][kk]["u50"],
                mode="lines+markers",
                name=models[k][kk]["name"],
                line=dict(
                    color=models[k][kk]["color"],
                    dash=models[k][kk]["dash"],
                ),
                marker=dict(
                    color=models[k][kk]["color"],
                    symbol=models[k][kk]["symbol"],
                ),
                legendgroup=kk,
            ),
            row=1, col=1,
        )
        # Add shadowed area corresponding to 25-75 (i.e., interquartile range)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(plot_data[k][kk]["u50"])),
                y=plot_data[k][kk]["u25"],
                mode="lines",
                line=dict(
                    color=models[k][kk]["color"],
                    width=1,
                ),
                # opacity=0.8,
                legendgroup=kk,
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(plot_data[k][kk]["u50"])),
                y=plot_data[k][kk]["u75"],
                mode="lines",
                line=dict(
                    color=models[k][kk]["color"],
                    width=1,
                ),
                # opacity=0.8,
                fill="tonexty",
                fillcolor=models[k][kk]["fillcolor"],
                legendgroup=kk,
                showlegend=False,
            )
        )

        # # # SAMPLES
        # Sample median of average activation
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(plot_data[k][kk]["s50"])),
                y = plot_data[k][kk]["s50"],
                mode="lines+markers",
                name=models[k][kk]["name"],
                line=dict(
                    color=models[k][kk]["color"],
                    dash=models[k][kk]["dash"],
                ),
                marker=dict(
                    color=models[k][kk]["color"],
                    symbol=models[k][kk]["symbol"],
                ),
                legendgroup=kk,
                showlegend=False,
            ),
            row=1, col=2,
        )
        # Add shadowed area corresponding to 25-75 (i.e., interquartile range)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(plot_data[k][kk]["s50"])),
                y=plot_data[k][kk]["s25"],
                mode="lines",
                line=dict(
                    color=models[k][kk]["color"],
                    width=1,
                ),
                # opacity=0.1,
                legendgroup=kk,
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(plot_data[k][kk]["s50"])),
                y=plot_data[k][kk]["s75"],
                mode="lines",
                line=dict(
                    color=models[k][kk]["color"],
                    width=1,
                ),
                # opacity=0.1,
                fill="tonexty",
                fillcolor=models[k][kk]["fillcolor"],
                legendgroup=kk,
                showlegend=False,
            ),
            row=1, col=2
        )


# Style figure

# Set subplot title font
fig.layout.annotations[0].font = dict(size=18, family="Times New Roman")
fig.layout.annotations[1].font = dict(size=18, family="Times New Roman")

# Set font and axis titles
fig.update_xaxes(
    title=dict(
        text=r"Epoch",
        font=dict(size=18, family="Times New Roman")),
    range=[0, 20],
    row=1,
    col=1,
    tickfont=dict(size=12, family="Times New Roman"),
)
fig.update_xaxes(
    title=dict(
        text=r"Epoch",
        font=dict(size=18, family="Times New Roman")),
    range=[0, 20],
    row=1,
    col=2,
    tickfont=dict(size=12, family="Times New Roman"),
)
fig.update_yaxes(
    title=dict(
        text=r"Average activity",
        font=dict(size=18, family="Times New Roman")),
    range=[0, 0.16],
    row=1,
    col=1,
    tickfont=dict(size=12, family="Times New Roman"),
)

fig.update_yaxes(
    title=dict(
        text=r"Average activity",
        font=dict(size=18, family="Times New Roman")),
    range=[0, 0.16],
    row=1,
    col=2,
    tickfont=dict(size=12, family="Times New Roman"),
)


fig.update_layout(
    # legend=dict(x=0.01, y=0.99),
    **fig_layout_double_subplot,
    template="plotly_white",
    showlegend=True,
)


fig.show()

# Savfe figure to PDF
fig.write_image(os.path.join(result_folder, "training_log.png"), scale=3, **fig_layout_double_subplot)