import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "3"

import jax
import jax.numpy as np
import numpy as onp
from tbparse import SummaryReader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from sklearn.svm import LinearSVC
from functools import partial
import yaml
import plotly.io as pio

np.set_printoptions(precision=4, suppress=True)
pio.renderers.default = "browser"

def logand_name(eps: str = ""):
    return r"$\mathcal{L}_{\epsilon}$"
    return r"$\text{MI} \left( \log \langle y_1, y_2 \rangle + \epsilon \right)$ " + eps

def and_name(alpha: str = ""):
    return r"$\mathcal{L}_{\alpha}$"
    return r"$\text{MI} \left( \langle y_1, y_2 \rangle \right) + \alpha L_1(y_1)$ " + alpha

models = {
    "vgg_gavg_sigmoid_logand": {
        "1": {"name": r"$\epsilon = 1e-2$", "chkp": 250, "color": "blue", "dash": "solid", "symbol":"circle"},
        "3": {"name": r"$\epsilon = 1e-1$", "chkp": 230, "color": "blue", "dash": "solid", "symbol":"circle"},
        "2": {"name": r"$\epsilon = 1e0$", "chkp": 250, "color": "blue", "dash": "solid", "symbol":"circle"},
    },
    "vgg_gavg_sigmoid_and": {
        "3": {"name": r"$\alpha = 8.0$", "chkp": 250, "color": "blue", "dash": "dash", "symbol":"x"},
        "2_bis": {"name": r"$\alpha = 4.0$", "chkp": 250, "color": "blue", "dash": "dash", "symbol":"x"},
        "1": {"name": r"$\alpha = 1.0$", "chkp": 250, "color": "blue", "dash": "dash", "symbol":"x"},
    },
}

print(logand_name(r"$\epsilon = 1e-2$"))

# base folder
base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    os.pardir,
)
base_folder = os.path.normpath(base_folder)

save_folder = os.path.join(
    base_folder,
    "resources",
    "results",
    "cifar10",
)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


fig_layout_double_subplot = {
    "height": 500,
    "width": 1000,
}
fig_layout_single = {
    "height": 500,
    "width": 700,   
}


k_types_and_subtitles = (
    ("vgg_gavg_sigmoid_logand", "vgg_gavg_sigmoid_and"),
    (logand_name(), and_name()),
)

# path to the file with stored result for accuracy, computing them everytime takes a lot of time
accuracy_results_path = os.path.join(
    base_folder,
    "resources",
    "results",
    "cifar10",
    "acc_vs_sparsity.yaml",
)

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
            "cifar10",
            k,
            kk,
            "activations",
            f"activations_chkp_{models[k][kk]['chkp']:03d}.npz",
        )

        data = onp.load(file_path)

        models[k][kk]["data"] = {k : data[k] for k in data.files}

        n_models += 1


"""---------------------"""
""" Plot PER-UNIT distribution of average activity """
"""---------------------"""


# First, compute manual bins for the histograms
# Indeed, a lot of units in the "and" models are exactly or almost 0, so we would like more bins there
per_unit_activity_bins = {}
low_exponent = -4
max_bin = 0
min_bin_center = (10**low_exponent)*0.7
for i, k in enumerate(k_types_and_subtitles[0]):
    per_unit_activity_bins[k] = {}
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=0)

        # take the maximum avg_act
        max_avg_act = np.max(avg_act)

        # create 50 bins in logarithmic scale from 1e-4 to max_avg_act
        bins = np.logspace(low_exponent, np.log10(max_avg_act+1e-4), 50)
        # pre-pend an initial 0
        bins = np.concatenate((np.zeros(1), bins))

        per_unit_activity_bins[k][kk] = bins

        if max_avg_act > max_bin:
            max_bin = max_avg_act+1e-4


fig = make_subplots(
    rows=1, cols=2, subplot_titles=k_types_and_subtitles[1], shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.1, vertical_spacing=0.15,
)

for i, k in enumerate(k_types_and_subtitles[0]):
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=0)

        print(avg_act.shape)

        # fig.add_trace(
        #     go.Histogram(
        #         x=avg_act,
        #         name=models[k][kk]["name"],
        #         opacity=0.5,
        #         # nbinsx=100,
        #         # histnorm="probability",
        #         # marker_color=models[k][kk]["color"],
        #         legend=f"legend{i+1}",
        #         xbins=dict(
        #             start=per_unit_activity_bins[k][kk][0],
        #             end=per_unit_activity_bins[k][kk][-1],
        #             size=(per_unit_activity_bins[k][kk][1:] - per_unit_activity_bins[k][kk][:-1]),
        #         ),
        #     ),
        #     row=1,
        #     col=i+1,
        # )
        
        
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
                # marker_color=models[k][kk]["color"],
                legend=f"legend{i+1}",
                width=(edges[1:] - edges[:-1]),
            ),
            row=1,
            col=i+1,
        )            

    # Set subplot title font
    fig.layout.annotations[i].font = dict(size=18, family="Times New Roman")
    # Set font and axis titles
    fig.update_xaxes(
        title=dict(
            text=r"Average activity (log-scale)",
            font=dict(size=18, family="Times New Roman")),
        row=1,
        col=i+1,
        tickfont=dict(size=16, family="Times New Roman"),
        type="log",
        # set range explicitly
        range=[np.log10(min_bin_center), np.log10(1.01)],
        # add ticks at powers of 10
        tickvals=[10**i for i in range(low_exponent, 1)],
    )
    fig.update_yaxes(
        title=dict(
            text=r"Frequency",
            font=dict(size=18, family="Times New Roman")),
        row=1,
        col=i+1,
        tickfont=dict(size=16, family="Times New Roman"),
    )
    # position the legend for this subplot
    fig.update_layout(
        **{f"legend{i+1}":dict(x=0.2+0.5*i, y=0.98, xanchor='right', yanchor='top'),}
    )

fig.update_layout(
    # title_text=r"Per-unit distribution of average activity",
    barmode="overlay",
    # legend=dict(x=0.01, y=0.99),
    **fig_layout_double_subplot,
    template="plotly_white",
)
fig.show()

# save figure as pdf
fig.write_image(os.path.join(save_folder, "per_unit_activity.pdf"), format="pdf", scale=3, **fig_layout_double_subplot)

# exit()
"""---------------------"""
""" Plot PER-SAMPLE distribution of average activity """
"""---------------------"""

# First, compute the maximum, so we know how to set axis range for both plot (so they have same range)


max_activation_per_sample = 0
for i, k in enumerate(k_types_and_subtitles[0]):
    per_unit_activity_bins[k] = {}
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=-1)
        max_avg_act = np.max(avg_act)

        if max_avg_act > max_activation_per_sample:
            max_activation_per_sample = max_avg_act+1e-4


fig = make_subplots(
    rows=1, cols=2, subplot_titles=k_types_and_subtitles[1], shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.1, vertical_spacing=0.15,
)

for i, k in enumerate(k_types_and_subtitles[0]):
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=-1)

        print(avg_act.shape)

        fig.add_trace(
            go.Histogram(
                x=avg_act,
                name=models[k][kk]["name"],
                opacity=0.5,
                nbinsx=50,
                histnorm="probability",
                # marker_color=models[k][kk]["color"],
                legend=f"legend{i+1}"
            ),
            row=1,
            col=i+1,
        )

    # Set subplot title font
    fig.layout.annotations[i].font = dict(size=18, family="Times New Roman")
    # Set font and axis titles
    fig.update_xaxes(
        title=dict(
            text=r"Average activity",
            font=dict(size=18, family="Times New Roman")),
        row=1,
        col=i+1,
        tickfont=dict(size=16, family="Times New Roman"),
        # set range manually
        range=[0.0, max_activation_per_sample],
    )
    fig.update_yaxes(
        title=dict(
            text=r"Frequency",
            font=dict(size=18, family="Times New Roman")),
        row=1,
        col=i+1,
        tickfont=dict(size=16, family="Times New Roman"),
    )
    # position the legend for this subplot
    fig.update_layout(
        **{f"legend{i+1}":dict(x=0.2+0.5*i, y=0.98, xanchor='right', yanchor='top'),}
    )

fig.update_layout(
    # title_text=r"Per-unit distribution of average activity",
    barmode="overlay",
    **fig_layout_double_subplot,
    template="plotly_white",
)
fig.show()

# save figure as pdf
fig.write_image(os.path.join(save_folder, "per_sample_activity.pdf"), format="pdf", scale=3, **fig_layout_double_subplot)


"""---------------------"""
""" Classification accuracy with varying level of sparsification """
"""---------------------"""


def keep_top_k(x, k):
        """ Keep only the k highest activations per sample, set rest to 0 """
        # x: (n_samples, n_features)
        # output: (n_samples, n_features)
        def keep_top_k_single(x_single, k):
            thresh = np.partition(x_single, -k)[-k]
            return np.where(x_single >= thresh, x_single, 0.0)
        # vmap over samples
        return jax.vmap(partial(keep_top_k_single, k=k))(x)

# first, check if a file with the stored results is already present
if os.path.exists(accuracy_results_path):
    
    print(f"Loading results from {accuracy_results_path}")
    with open(accuracy_results_path, "r") as f:
        accuracy_results = yaml.safe_load(f)

    # Add it to the models dict
    N_K = accuracy_results["n_top_k"]
    for k in models.keys():
        for kk in models[k].keys():
            models[k][kk]["svm_acc"] = {}
            for i_top_k, n_top_k in enumerate(N_K):
                models[k][kk]["svm_acc"][n_top_k] = {
                    "train": accuracy_results[k][kk]["train"][i_top_k],
                    "val": accuracy_results[k][kk]["val"][i_top_k],
                }
else:

    N_K = [256, 25, 20, 15, 10, 5]

    # For each model, train a linear SVM after sparsifying the activations by keeping only the k highest activations per sample

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
for i, k in enumerate(models.keys()):
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
                legend=f"legend{i+1}"
            )
        )

    # place the legend
    fig.update_layout(
        **{f"legend{i+1}":dict(x=1.1, y=(0.98 - i*0.3), xanchor='left', yanchor='top', title_text=k_types_and_subtitles[1][i]),}
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
    **fig_layout_single,
    template="plotly_white",
)

fig.show()


# save figure as pdf
fig.write_image(os.path.join(save_folder, "accuracy_vs_sparsity.pdf"), format="pdf", scale=3, **fig_layout_single)