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

np.set_printoptions(precision=4, suppress=True)

def logand_name(eps: str = ""):
    return r"$\text{MI} \left( \log \langle y_1, y_2 \rangle + \epsilon \right)$ " + eps

def and_name(alpha: str = ""):
    return r"$\text{MI} \left( \langle y_1, y_2 \rangle \right) + \alpha L_1(y_1)$ " + alpha

models = {
    "vgg_gavg_sigmoid_logand": {
        "1": {"name": r"$\epsilon = 1e-2$", "chkp": 250, "color": "blue", "dash": "solid", "symbol":"circle"},
        "2": {"name": r"$\epsilon = 1e0$", "chkp": 250, "color": "blue", "dash": "solid", "symbol":"circle"},
    },
    "vgg_gavg_sigmoid_and": {
        "1": {"name": r"$\alpha = 1.0$", "chkp": 250, "color": "blue", "dash": "dash", "symbol":"x"},
        "2_bis": {"name": r"$\alpha = 4.0$", "chkp": 250, "color": "blue", "dash": "dash", "symbol":"x"},
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

k_types_and_subtitles = (
    ("vgg_gavg_sigmoid_logand", "vgg_gavg_sigmoid_and"),
    (logand_name(), and_name()),
)


fig = make_subplots(
    rows=1, cols=2, subplot_titles=k_types_and_subtitles[1], shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.1, vertical_spacing=0.15,
)

for i, k in enumerate(k_types_and_subtitles[0]):
    for kk in models[k].keys():
        
        avg_act = models[k][kk]["data"]["zs"].mean(axis=0)

        print(avg_act.shape)

        fig.add_trace(
            go.Histogram(
                x=avg_act,
                name=models[k][kk]["name"],
                opacity=0.5,
                nbinsx=50,
                histnorm="probability",
                # marker_color=models[k][kk]["color"],
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
    )
    fig.update_yaxes(
        title=dict(
            text=r"Frequency",
            font=dict(size=18, family="Times New Roman")),
        row=1,
        col=i+1,
        tickfont=dict(size=16, family="Times New Roman"),
    )

fig.update_layout(
    # title_text=r"Per-unit distribution of average activity",
    barmode="overlay",
    legend=dict(x=0.01, y=0.99),
    width=1200,
    height=600,
    template="plotly_white",
)
fig.show()



"""---------------------"""
""" Plot PER-SAMPLE distribution of average activity """
"""---------------------"""


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
    )
    fig.update_yaxes(
        title=dict(
            text=r"Frequency",
            font=dict(size=18, family="Times New Roman")),
        row=1,
        col=i+1,
        tickfont=dict(size=16, family="Times New Roman"),
    )

fig.update_layout(
    # title_text=r"Per-unit distribution of average activity",
    barmode="overlay",
    legend=dict(x=0.01, y=0.99),
    width=1200,
    height=600,
    template="plotly_white",
)
fig.show()


"""---------------------"""
""" Classification accuracy with varying level of sparsification """
"""---------------------"""

N_K = [5, 10, 15, 20, 25]

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

    keep_top_k_jitted = jax.jit(partial(keep_top_k, k=n_top_k))

    for k in tqdm(models.keys(), total=len(models.keys()), leave=False):
        for kk in tqdm(models[k].keys(), total=len(models[k].keys()), leave=False):
            
            # take_first = 5000  # to speed up the process for debugging
            
            zs = models[k][kk]["data"]["zs"].copy()#[:take_first]
            zs_val = models[k][kk]["data"]["zs_val"].copy()#[:take_first]
            labels_onehot = models[k][kk]["data"]["labels_onehot"].copy()#[:take_first]
            labels_onehot_val = models[k][kk]["data"]["labels_onehot_val"].copy()#[:take_first]

            labels_categorical = labels_onehot.argmax(axis=-1)
            labels_categorical_val = labels_onehot_val.argmax(axis=-1)

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

for k in models.keys():
    for kk in models[k].keys():
        
        accs_val = [models[k][kk]["svm_acc"][n_top_k]["val"] for n_top_k in N_K]

        fig.add_trace(
            go.Scatter(
                x=N_K,
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
        text=r"Non-zero features ($k$)",
        font=dict(size=18, family="Times New Roman")),
    tickfont=dict(size=16, family="Times New Roman"),
    dtick=5,
)

fig.update_yaxes(
    title=dict(
        text=r"Accuracy",
        font=dict(size=18, family="Times New Roman")),
    tickfont=dict(size=16, family="Times New Roman"),
    # range=[0.7, 0.92],
)

fig.update_layout(
    title_text=r"Classification accuracy vs. sparsification",
    legend=dict(x=0.01, y=0.99),
    width=700,
    height=500,
    template="plotly_white",
)

fig.show()