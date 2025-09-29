import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "0"

import jax
import jax.numpy as np
import numpy as onp
# from tbparse import SummaryReader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

np.set_printoptions(precision=4, suppress=True)
pio.renderers.default = "browser"


models = {
    "standard": {
        "1": {"name": r"$p^* = 0.05$", "chkp": 20, "color": "blue"},
    },
}

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
)
base_folder = os.path.normpath(base_folder)


"""---------------------"""
""" Import activation data """
"""---------------------"""

n_models = 0

for k in models.keys():
    for kk in models[k].keys():
        print(f"{k} - {kk}")

        file_path = os.path.join(
            base_folder,
            "resources",
            "models",
            "antihebbian",
            k,
            kk,
            "activations",
            f"activations_chkp_{models[k][kk]['chkp']:03d}.npz",
        )

        data = onp.load(file_path)

        models[k][kk]["data"] = {k : data[k] for k in data.files}

        n_models += 1


"""---------------------"""
""" Plot PER-UNIT and PER-SAMPLE distribution of average activity """
"""---------------------"""

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Units",
        "Samples",
    ),
)

for i, k in enumerate(models.keys()):
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=0)

        print(avg_act.shape)

        fig.add_trace(
            go.Histogram(
                x=avg_act,
                name=models[k][kk]["name"],
                # opacity=0.5,
                nbinsx=50,
                histnorm="probability",
                marker_color=models[k][kk]["color"],
            ),
            row=1, col=1
        )

for i, k in enumerate(models.keys()):
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=-1)

        print(avg_act.shape)

        fig.add_trace(
            go.Histogram(
                x=avg_act,
                name=models[k][kk]["name"],
                opacity=0.5,
                # nbinsx=50,
                histnorm="probability",
                marker_color=models[k][kk]["color"],
            ),
            row=1, col=2
        )

for i in range(2):
    # Set font and axis titles
    fig.update_xaxes(
        title=dict(
            text=r"Average activity",
            font=dict(size=18, family="Times New Roman")),
        tickfont=dict(size=16, family="Times New Roman"),
        row=1, col=i+1
    )
    fig.update_yaxes(
        title=dict(
            text=r"Frequency",
            font=dict(size=18, family="Times New Roman")),
        tickfont=dict(size=16, family="Times New Roman"),
        row=1, col=i+1
    )

fig.update_layout(
    # title_text=r"Per-unit distribution of average activity",
    barmode="overlay",
    legend=dict(x=0.01, y=0.99),
    width=1200,
    height=600,
    template="plotly_white",
    showlegend=False,
)
fig.show()