import os
import jax
import jax.numpy as np
from jax import jit, grad, vmap
import sbdr
import json
import optax
import gzip
import pickle
from functools import partial
import flax.linen as nn

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

np.set_printoptions(precision=4, suppress=True)

SAVE = False

run_name = "test"

n_epochs = 200
n_batch_samples = 128
learning_rate = 0.001
n_epochs_save = 3

normalize_min = -1.0
normalize_max = 1.0

"""------------------"""
""" Model params """
"""------------------"""

model_folder_name = "first_test"


# model_info_dict = {}

# model_info_dict["model_name"] = model_name

# model_info_dict["n_features"] = n_features
# model_info_dict["p_target"] = p_target
# model_info_dict["momentum"] = momentum
# model_info_dict["init_variance_q"] = init_variance_q
# model_info_dict["init_variance_w"] = init_variance_w
# model_info_dict["lr_scale_w"] = lr_scale_w


# params_base_path = os.path.join(
#     os.path.dirname(__file__),
#     os.path.pardir,
#     os.path.pardir,
#     "resources",
#     "trained_models",
#     "fooldiak",
# )

# if not os.path.exists(params_base_path):
#     os.makedirs(params_base_path)

# key = jax.random.PRNGKey(0)  # jax.random.PRNGKey(time.time_ns()) #

# print(f"\nModel: '{model_name}'")
# print(f"\tOutput Features: {n_features}")
# print(f"\tTarget Activation: {p_target} ({np.rint(p_target*n_features)} units)")
# print(f"\tMomentum: {momentum}")


"""------------------"""
""" Import Dataset """
"""------------------"""

data_folder_path = os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir,
    "resources",
    "datasets",
    "fashion-mnist",
    "data",
    "fashion",
)

print("\nLoading Dataset")


def load_mnist(path, kind="train"):
    """Load FashionMNIST data from `path`"""
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


X, Y = load_mnist(data_folder_path, "train")
test_X, test_Y = load_mnist(data_folder_path, "t10k")

print(f"\tLoaded dataset from: {data_folder_path}")
print(f"\tOriginal Shapes:")
print(f"\t\tX: {X.shape}")
print(f"\t\tY: {Y.shape}")
print(f"\t\ttest_X: {test_X.shape}")
print(f"\t\ttest_Y: {test_Y.shape}")

print(X.min(), X.max())
print(test_X.min(), test_X.max())

exit()

X = (X - X.min()) / (X.max() - X.min())
X = X * (normalize_max - normalize_min) + normalize_min

test_X = (test_X - test_X.min()) / (test_X.max() - test_X.min())
test_X = test_X * (normalize_max - normalize_min) + normalize_min


# Shuffle using the same key, so that the labels are shuffled in the same way as the images
key = jax.random.PRNGKey(SHUFFLE_SEED)
key, subkey = jax.random.split(key)

# Shuffle the training set
X = jax.random.permutation(key, X, axis=0, independent=False)
Y = jax.random.permutation(key, Y, axis=0, independent=False)
# Shuffle the test set
test_X = jax.random.permutation(subkey, test_X, axis=0, independent=False)
test_Y = jax.random.permutation(subkey, test_Y, axis=0, independent=False)

# # Reshape to add the channel dimension
# X = X.reshape((X.shape[0], 28, 28, 1))
# test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

model_info_dict["dataset"] = {}
model_info_dict["dataset"]["path"] = data_base_path
model_info_dict["dataset"]["min"] = float(X.min())
model_info_dict["dataset"]["max"] = float(X.max())
model_info_dict["dataset"]["single_input_shape"] = list(X[0].shape)

print(f"\tFinal Shapes:")
print(f"\t\tX: {X.shape}")
print(f"\t\tY: {Y.shape}")
print(f"\t\ttest_X: {test_X.shape}")
print(f"\t\ttest_Y: {test_Y.shape}")


"""------------------"""
""" Try to import parameters from model which was already trained, otherwise initialize new parameters. """
"""------------------"""

PARAM_SEED = 123

print("\nInitializing Model")

model = sdm.FoldiakLayer(
    n_features=n_features,
    p_target=p_target,
    momentum=momentum,
    init_variance_q=init_variance_q,
    init_variance_w=init_variance_w,
    lr_scale_w=lr_scale_w,
)

print(f"\tsingle input size: {X[0].shape}")

filepath = os.path.join(params_base_path, model_name)

if os.path.exists(os.path.join(params_base_path, model_name + "_params.pkl")):
    print("Saved parameters found - loading from file")
    with open(os.path.join(params_base_path, model_name + "_params.pkl"), "rb") as f:
        params = pickle.load(f)
else:
    print("No existing parameters - initializing new")
    key_params = jax.random.key(PARAM_SEED)
    params = model.init(key_params, X[:3])

model_info_dict["params"] = {}
model_info_dict["params"]["seed"] = PARAM_SEED

print(f"\tDict of params: \n\t{params['params'].keys()}")

# print_pytree_shapes(params)


"""------------------"""
""" Try one forward pass """
"""------------------"""


def apply_model(params, xs, model):
    return model.apply(params, xs)


apply_model = partial(apply_model, model=model)
apply_model = jit(apply_model)

print("\nTest one forward pass")

xs = X[:n_batch_samples]
zs = apply_model(params, xs)

print(f"\tOutput Info:")
print(f"\t\tshape: {zs.shape}")
print(f"\t\tmean active units: {zs.sum(axis=-1).mean()}")
print(f"\t\tstd active units: {zs.sum(axis=-1).std()}")

# exit("AAAAAA")


"""------------------"""
""" Weight Update """
"""------------------"""


def update_params(params, xs, lr, model):
    zs = model.apply(params, xs)
    params = model.update_params(params, xs, zs, lr)
    return params, zs


update_params = partial(update_params, model=model)
update_params = jit(update_params)


print("\nTest 50 weight updates")

xs = X[:n_batch_samples]
tmp__params = params.copy()
for i in range(50):
    tmp__params, zs = update_params(tmp__params, xs, lr=0.1)

print("\t Done.")

# exit("OOAIAOIA")

"""------------------"""
""" Statistics """
"""------------------"""

# quantiles for which we compute activation during training
qs = np.array((0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95))


# compute the statistics of a batch of activation
def compute_activity_stats_gen(per_sample_activity, qs):
    """per_sample_activity is assumed to have shape (batch,)"""
    mean = per_sample_activity.mean()
    std = per_sample_activity.std()
    # quantiles
    qs = np.quantile(per_sample_activity, qs)

    return mean, std, qs


def compute_unit_stats_gen(per_unit_activity, qs):
    """per_unit_activity is assumed to have shape (feature, )"""
    mean = per_unit_activity.mean()
    std = per_unit_activity.std()
    # quantiles
    qs = np.quantile(per_unit_activity, qs)

    return mean, std, qs


def compute_zs_stats_gen(zs, qs):
    """zs is assumed to have shape (batch, feature)"""
    per_sample_activity = zs.sum(axis=-1)
    mean_act, std_act, qs_act = compute_activity_stats_gen(per_sample_activity, qs)

    per_unit_activity = zs.mean(axis=0)
    mean_unit, std_unit, qs_unit = compute_unit_stats_gen(per_unit_activity, qs)

    return (mean_act, std_act, qs_act), (mean_unit, std_unit, qs_unit)


compute_activity_stats = jit(partial(compute_activity_stats_gen, qs=qs))

compute_zs_stats = jit(partial(compute_zs_stats_gen, qs=qs))

history = {
    "epoch_stats": {
        "train": {
            "mean": [],
            "std": [],
            "qs": [],
        },
        "test": {
            "mean": [],
            "std": [],
            "qs": [],
        },
    },
    "unit_stats": {
        "train": {
            "mean": [],
            "std": [],
            "qs": [],
        },
        "test": {
            "mean": [],
            "std": [],
            "qs": [],
        },
    },
}

# COmpute statistics before training, for both train and test
tmp = apply_model(params, X)
(mean_act, std_act, qs_act), (mean_unit, std_unit, qs_unit) = compute_zs_stats(tmp)
history["epoch_stats"]["train"]["mean"].append(mean_act)
history["epoch_stats"]["train"]["std"].append(std_act)
history["epoch_stats"]["train"]["qs"].append(qs_act)
history["unit_stats"]["train"]["mean"].append(mean_unit)
history["unit_stats"]["train"]["std"].append(std_unit)
history["unit_stats"]["train"]["qs"].append(qs_unit)

tmp = apply_model(params, test_X)
(mean_act, std_act, qs_act), (mean_unit, std_unit, qs_unit) = compute_zs_stats(tmp)
history["epoch_stats"]["test"]["mean"].append(mean_act)
history["epoch_stats"]["test"]["std"].append(std_act)
history["epoch_stats"]["test"]["qs"].append(qs_act)
history["unit_stats"]["test"]["mean"].append(mean_unit)
history["unit_stats"]["test"]["std"].append(std_unit)
history["unit_stats"]["test"]["qs"].append(qs_unit)


"""------------------"""
""" Training """
"""------------------"""

n_print_batch = 25

print("\nStart Training")

# Training Loop
try:
    for ne in range(n_epochs):
        print(f"Epoch: {ne:03d}/{n_epochs:03d}")

        # shuffle the dataset (and the labels, with the same key)
        key, _ = jax.random.split(key)
        idx = jax.random.permutation(
            key, np.arange(X.shape[0]), axis=0, independent=False
        )
        X = X[idx]
        Y = Y[idx]

        # train in minibatches
        for nb in range(int(X.shape[0] / n_batch_samples)):

            # select inputs for this batch
            xs = X[nb * n_batch_samples : (nb + 1) * n_batch_samples]

            # # Apply Salt and Pepper Noise
            # noise_amount = 0.05
            # key, _ = jax.random.split(key)
            # key_p, key_n = jax.random.split(key)
            # noise_p = jax.random.bernoulli(key_p, noise_amount, shape=xs.shape)
            # noise_p = noise_p * (normalize_max - normalize_min) + normalize_min
            # noise_n = jax.random.bernoulli(key_n, noise_amount, shape=xs.shape)
            # noise_n = noise_n * (normalize_max - normalize_min) + normalize_min
            # xs = np.clip(xs + noise_p - noise_n, normalize_min, normalize_max)

            # # Apply Gaussian Noise
            # noise_amount = 0.2
            # key, _ = jax.random.split(key)
            # noise_normal = noise_amount * \
            #     jax.random.normal(key, shape=xs.shape)
            # xs = np.clip(xs + noise_normal, normalize_min, normalize_max)

            # Update parameters
            params, zs = update_params(params, xs, lr=learning_rate)

        # save the weights every n_epochs_save epochs
        if SAVE and ((ne % n_epochs_save == 0) and (ne > 0)):
            # Save model
            filepath = os.path.join(params_base_path, model_name)
            with open(filepath + "_params.pkl", "wb") as f:
                pickle.dump(params, f)
            # Save dict of informations
            with open(filepath + "_info.json", "w") as f:
                json.dump(
                    model_info_dict,
                    f,
                )
            print("\tModel saved.")

        # # Compute statistics for the whole epoch
        # Training Set
        tmp = apply_model(params, X)
        (mean_act, std_act, qs_act), (mean_unit, std_unit, qs_unit) = compute_zs_stats(
            tmp
        )
        history["epoch_stats"]["train"]["mean"].append(mean_act)
        history["epoch_stats"]["train"]["std"].append(std_act)
        history["epoch_stats"]["train"]["qs"].append(qs_act)
        history["unit_stats"]["train"]["mean"].append(mean_unit)
        history["unit_stats"]["train"]["std"].append(std_unit)
        history["unit_stats"]["train"]["qs"].append(qs_unit)

        # print the statistics on the training set
        print("\tTrain")
        print(
            f"\t\tA: mean:{history['epoch_stats']['train']['mean'][-1]:.3f} qs:{history['epoch_stats']['train']['qs'][-1]}"
        )
        print(
            f"\t\tU: mean:{history['unit_stats']['train']['mean'][-1]:.3f} qs:{history['unit_stats']['train']['qs'][-1]}"
        )

        # Test Set
        tmp = apply_model(params, test_X)
        # compute the statistics
        (mean_act, std_act, qs_act), (mean_unit, std_unit, qs_unit) = compute_zs_stats(
            tmp
        )
        history["epoch_stats"]["test"]["mean"].append(mean_act)
        history["epoch_stats"]["test"]["std"].append(std_act)
        history["epoch_stats"]["test"]["qs"].append(qs_act)
        history["unit_stats"]["test"]["mean"].append(mean_unit)
        history["unit_stats"]["test"]["std"].append(std_unit)
        history["unit_stats"]["test"]["qs"].append(qs_unit)

        # print the statistics on the training set
        print("\tTest")
        print(
            f"\t\tA: mean:{history['epoch_stats']['test']['mean'][-1]:.3f} qs:{history['epoch_stats']['test']['qs'][-1]}"
        )
        print(
            f"\t\tU: mean:{history['unit_stats']['test']['mean'][-1]:.3f} qs:{history['unit_stats']['test']['qs'][-1]}"
        )


except KeyboardInterrupt:
    pass


""" Save model's params """

if SAVE:
    # Save model, to be used for testing of Temporal Learning
    filepath = os.path.join(params_base_path, model_name)
    with open(filepath + "_params.pkl", "wb") as f:
        pickle.dump(params, f)
    # save also information about this model
    with open(filepath + "_info.json", "w") as f:
        json.dump(
            model_info_dict,
            f,
        )

    print("\nModel saved.")


# exit("XNWLKJ")


"""------------------"""
""" Plot some statistics of unit activation """
"""------------------"""

for k in history["epoch_stats"]["test"].keys():
    history["epoch_stats"]["test"][k] = np.array(history["epoch_stats"]["test"][k])

for k in history["epoch_stats"]["train"].keys():
    history["epoch_stats"]["train"][k] = np.array(history["epoch_stats"]["train"][k])

# print(history["epoch_stats"]["test"].keys())
# print(history["epoch_stats"]["test"]["mean"].shape)
# print(history["epoch_stats"]["test"]["qs"].shape)

# exit("UGABUGA")

# plot history of mean active units and shaded areas of quantiles, both for train and test


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["train"]["mean"].shape[0]),
        y=history["epoch_stats"]["train"]["mean"],
        mode="lines",
        name="train mean",
        legendgroup="train",
        line=dict(color="blue"),
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["test"]["mean"].shape[0]),
        y=history["epoch_stats"]["test"]["mean"],
        mode="lines",
        name="test mean",
        legendgroup="test",
        line=dict(color="red"),
    )
)

# add shadowed area showing 25%-75% bounds

# train 25
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["train"]["mean"].shape[0]),
        y=history["epoch_stats"]["train"]["qs"][:, 1],
        mode="lines",
        name=f"train {qs[1]}-{qs[-2]}",
        fill=None,
        line=dict(color="blue", width=0),
        legendgroup="train",
    )
)
# train 75
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["train"]["mean"].shape[0]),
        y=history["epoch_stats"]["train"]["qs"][:, -2],
        mode="lines",
        name=f"train {qs[1]}-{qs[-2]}",
        fill="tonexty",
        # put some transparency
        fillcolor="rgba(0,0,255,0.2)",
        legendgroup="train",
    )
)

# test 25
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["test"]["mean"].shape[0]),
        y=history["epoch_stats"]["test"]["qs"][:, 1],
        mode="lines",
        name=f"test {qs[1]}-{qs[-2]}",
        fill=None,
        line=dict(color="red", width=0),
        legendgroup="test",
    )
)
# test 75
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["test"]["mean"].shape[0]),
        y=history["epoch_stats"]["test"]["qs"][:, -2],
        mode="lines",
        name=f"test {qs[1]}-{qs[-2]}",
        fill="tonexty",
        # put some transparency
        fillcolor="rgba(255,0,0,0.2)",
        legendgroup="test",
    )
)

# add shadowed area showing 5%-95% bounds

# train 5
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["train"]["mean"].shape[0]),
        y=history["epoch_stats"]["train"]["qs"][:, 0],
        mode="lines",
        name=f"train {qs[0]}-{qs[-1]}",
        fill=None,
        line=dict(color="blue", width=0),
        legendgroup="train",
    )
)
# train 95
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["train"]["mean"].shape[0]),
        y=history["epoch_stats"]["train"]["qs"][:, -1],
        mode="lines",
        name=f"train {qs[0]}-{qs[-1]}",
        fill="tonexty",
        # put some transparency
        fillcolor="rgba(0,0,255,0.1)",
        legendgroup="train",
    )
)

# test 5
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["test"]["mean"].shape[0]),
        y=history["epoch_stats"]["test"]["qs"][:, 0],
        mode="lines",
        name=f"test {qs[0]}-{qs[-1]}",
        fill=None,
        line=dict(color="red", width=0),
        legendgroup="test",
    )
)
# test 95
fig.add_trace(
    go.Scatter(
        x=np.arange(history["epoch_stats"]["test"]["mean"].shape[0]),
        y=history["epoch_stats"]["test"]["qs"][:, 4],
        mode="lines",
        name=f"test {qs[0]}-{qs[-1]}",
        fill="tonexty",
        # put some transparency
        fillcolor="rgba(255,0,0,0.1)",
        legendgroup="test",
    )
)

fig.update_layout(
    title=f"Mean Active Units - {model_name}",
    xaxis_title="Epoch",
    yaxis_title="Mean Active Units",
    legend_title="Data",
)

fig.show()
