import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "0"

import jax
import jax.numpy as np
import numpy as onp
# from tbparse import SummaryReader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from tqdm import tqdm
from functools import partial
from sklearn.svm import LinearSVC

np.set_printoptions(precision=4, suppress=True)
pio.renderers.default = "browser"


models = {
    "standard": {
        "3": {"name": r"$p_* = 0.01$", "chkp": 20, "color": "#1f77b4", "dash": "solid", "symbol": "circle"},
        "2": {"name": r"$p_* = 0.02$", "chkp": 20, "color": "#be44ff", "dash": "solid", "symbol": "circle"},
        "1": {"name": r"$p_* = 0.05$", "chkp": 20, "color": "#2ca02c", "dash": "solid", "symbol": "circle"},
        "4": {"name": r"$p_* = 0.075$", "chkp": 20, "color": "#d62728", "dash": "solid", "symbol": "circle"},
    },
}

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
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
""" Compute PER-UNIT and PER-SAMPLE distribution of average activity """
"""---------------------"""

# First, compute manual bins for the histograms
# Indeed, a lot of units in the "and" models are exactly or almost 0, so we would like more bins there
per_unit_activity_bins = {}
per_sample_activity_bins = {}

low_exponent_unit = -3
low_exponent_sample = np.log10(1/256 - 1e-6).item()

max_bin = 0
min_bin_center_unit = (10**low_exponent_unit)*0.7
min_bin_center_sample = (10**low_exponent_sample)*0.7

for k in models.keys():
    per_unit_activity_bins[k] = {}
    per_sample_activity_bins[k] = {}
    for kk in models[k].keys():        

        avg_act_unit = models[k][kk]["data"]["zs"].mean(axis=0)
        tot_act_sample = models[k][kk]["data"]["zs"].sum(axis=1)

        # take the maximum avg_act
        max_avg_act_unit = np.max(avg_act_unit)
        max_tot_act_sample = np.max(tot_act_sample)

        # create 50 bins in logarithmic scale from 1e-4 to max_avg_act
        bins_unit = np.logspace(low_exponent_unit, np.log10(max_avg_act_unit+1e-4), 50)
        # for the sample activity, instead, we already know is discretized by integer value from 0 to max_tot_act_sample
        bins_sample = np.arange(1, 50, 1, dtype=np.float32)/256

        # pre-pend an initial 0
        bins_unit = np.concatenate((np.zeros(1), bins_unit))
        bins_sample = np.concatenate((np.zeros(1), bins_sample))

        per_unit_activity_bins[k][kk] = bins_unit
        per_sample_activity_bins[k][kk] = bins_sample

        if max_avg_act_unit > max_bin:
            max_bin = max_avg_act_unit+1e-4

        if max_tot_act_sample > max_bin:
            max_bin = max_tot_act_sample+1e-4

"""---------------------"""
""" Plot PER-UNIT and PER-SAMPLE distribution of average activity """
"""---------------------"""

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Units",
        "Samples",
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.15,
)

for i, k in enumerate(models.keys()):
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=0)

        edge_centers = 0.5 * (per_unit_activity_bins[k][kk][1:] + per_unit_activity_bins[k][kk][:-1])
        edges = per_unit_activity_bins[k][kk]
        counts, _ = np.histogram(avg_act, bins=edges)
        # normalize counts to sum to 1
        counts = counts / np.sum(counts)

        fig.add_trace(
            go.Bar(
                x=edge_centers,
                y=counts,
                name=models[k][kk]["name"],
                opacity=0.5,
                # nbinsx=100,
                # histnorm="probability",
                marker_color=models[k][kk]["color"],
                legend=f"legend{i+1}",
                width=(edges[1:] - edges[:-1]),
            ),
            row=1,
            col=1,
        )            


for i, k in enumerate(models.keys()):
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=-1)

        edge_centers = 0.5 * (per_sample_activity_bins[k][kk][1:] + per_sample_activity_bins[k][kk][:-1])
        edges = per_sample_activity_bins[k][kk]
        counts, _ = np.histogram(avg_act, bins=edges)
        # normalize counts to sum to 1
        counts = counts / np.sum(counts)

        fig.add_trace(
            go.Bar(
                x=edge_centers,
                y=counts,
                name=models[k][kk]["name"],
                opacity=0.5,
                # nbinsx=100,
                # histnorm="probability",
                marker_color=models[k][kk]["color"],
                legend=f"legend{i+1}",
                width=(edges[1:] - edges[:-1]),
            ),
            row=1,
            col=2,
        )   


# Set font and axis titles
for i, (min_bin_center, low_exponent) in enumerate(zip([min_bin_center_unit, min_bin_center_sample], [low_exponent_unit, low_exponent_sample])):
    fig.update_xaxes(
        title=dict(
            text=r"Average activity (log-scale)",
            font=dict(size=18, family="Times New Roman")),
        tickfont=dict(size=16, family="Times New Roman"),
        row=1,
        col=i+1,
        # log scale
        type="log",
        # set range explicitly
        range=[np.log10(min_bin_center_unit), np.log10(1.01)],
        # add ticks at powers of 10
        tickvals=[10**j for j in range(int(low_exponent), 1)],
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
    **fig_layout_double_subplot,
    template="plotly_white",
    showlegend=False,
)
fig.show()

# export image as PDF
fig.write_image(
    os.path.join(
        result_folder,
        "activation_stats.pdf"
    ),
    format="pdf",
    **fig_layout_double_subplot,
    scale=3,
)

exit()

"""---------------------"""
""" Classification accuracy with varying level of sparsification """
"""---------------------"""

N_K = [512, 50, 40, 30, 20, 10]

# For each model, train a linear SVM after sparsifying the activations by keeping only the k highest activations per sample

def keep_top_k(x, k):
    """ Keep only the k highest activations per sample, set rest to 0 """
    # x: (n_samples, n_features)
    # output: (n_samples, n_features)
    def keep_top_k_single(x_single, k):
        thresh = np.partition(x_single, -k)[-k]
        return np.where(x_single >= thresh, x_single, 0.0)
    # vmap over samples
    return jax.vmap(partial(keep_top_k_single, k=k))(x)


# NOTE, this nested cycles takes a while to run
for n_top_k in tqdm(N_K):

    sparsity = 1.0 - n_top_k/256.0
    keep_top_k_jitted = jax.jit(partial(keep_top_k, k=n_top_k))

    for k in tqdm(models.keys(), total=len(models.keys()), leave=False):
        for kk in tqdm(models[k].keys(), total=len(models[k].keys()), leave=False):
            
            take_first = 20  # to speed up the process for debugging
            
            zs = models[k][kk]["data"]["zs"].copy()#[:take_first]
            zs_val = models[k][kk]["data"]["zs_val"].copy()#[:take_first]
            labels_onehot = models[k][kk]["data"]["labels_onehot"].copy()#[:take_first]
            labels_onehot_val = models[k][kk]["data"]["labels_onehot_val"].copy()#[:take_first]

            labels_categorical = labels_onehot.argmax(axis=-1)
            labels_categorical_val = labels_onehot_val.argmax(axis=-1)

            if sparsity > 0.1:
                zs = keep_top_k_jitted(zs)
                zs_val = keep_top_k_jitted(zs_val)

            svm_model = LinearSVC(
                random_state=0,
                tol=1e-4,
                multi_class="ovr",
                intercept_scaling=1,
                C=8,
                penalty="l1",
                loss="squared_hinge",
                max_iter=2000,
            )

            svm_model.fit(
                zs,
                labels_categorical,
            )
            acc_train = svm_model.score(zs, labels_categorical)
            acc_val = svm_model.score(zs_val, labels_categorical_val)

            models[k][kk].setdefault("svm_acc", {})

            models[k][kk]["svm_acc"][n_top_k] = {
                "train": acc_train,
                "val": acc_val,
            }

for k in models.keys():
    for kk in models[k].keys():
        print(f"{k} - {kk}")
        print("n_top_k\ttrain\tval")
        for n_top_k in N_K:
            accs = models[k][kk]["svm_acc"][n_top_k]
            print(f"{n_top_k}\t{accs['train']:.4f}\t{accs['val']:.4f}")
        print()


"""---------------------"""
""" plot the accuracy as a function of n_top_k """
"""---------------------"""

fig = go.Figure()
sparsities = 1.0 - np.array(N_K)/256.0
log_sparsities = 25.0**sparsities
for k in models.keys():
    for kk in models[k].keys():
        
        accs_val = np.array([models[k][kk]["svm_acc"][n_top_k]["val"] for n_top_k in N_K])
        

        fig.add_trace(
            go.Scatter(
                x=log_sparsities,
                y=accs_val,
                mode="lines+markers",
                name=models[k][kk]["name"],
                line=dict(
                    # color=models[k][kk]["color"],
                    dash=models[k][kk]["dash"],
                ),
                marker=dict(
                    symbol=models[k][kk]["symbol"],
                    size=10,
                ),
            )
        )

# Set font and axis titles
fig.update_xaxes(
    title=dict(
        text="Non-zero features",
        font=dict(size=18, family="Times New Roman")),
    tickfont=dict(size=16, family="Times New Roman"),
    # place ticks at the values of sparsities
    tickvals=log_sparsities,
    ticktext=[f"{k}" for k in N_K],
)

fig.update_yaxes(
    title=dict(
        text="Accuracy (%)",
        font=dict(size=18, family="Times New Roman")),
    tickfont=dict(size=16, family="Times New Roman"),
    # range=[0.7, 0.92],
)

fig.update_layout(
    # title_text=r"Classification accuracy vs. sparsification",
    # legend=dict(x=0.01, y=0.99),
    width=1200,
    height=700,
    template="plotly_white",
)

fig.show()

# export image as PDF
fig.write_image(
    os.path.join(
        result_folder,
        "svm_accuracy_vs_sparsity.pdf"
    ),
    format="pdf"
)