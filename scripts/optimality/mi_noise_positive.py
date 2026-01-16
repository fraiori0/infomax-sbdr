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

# Online tool to visualize the pdf of the Beta distribution
# https://www.acsu.buffalo.edu/~adamcunn/probability/beta.html

if __name__ == "__main__":

    SAVE = True

    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "sampled_mi",
        "noise_positive",
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    N_SEEDS = 10
    SEEDS = 100 * np.arange(1, N_SEEDS + 1)

    # Number of features (i.e., dimension of a single sample)
    N_FEATURES = 256

    # Number of samples drawn from the gamma distribution
    N_GAMMA_SAMPLES = 100
    # Number of binary samples drawn from each gamma distribution
    N_SINGLE_MASK_SAMPLES = 100
    # Expected number of non-zero units after Bernoulli-sampling from N_FEATURES gamma-distributed probabilities over
    GAMMA_AVERAGE_NON_ZERO = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    # Concentration of the gamma distribution
    GAMMA_CONCENTRATION = 100.0
    # Range for the uniform sampling of the activation level, after selecting the non-zero units
    UNIFORM_RANGE = (0.9, 1.0)
    # Similarity bias
    EPS_SIM = 1e-4

    # values for the positive noise probability
    NOISE_PROBS = [0.0, 0.002, 0.004, 0.008, 0.01, 0.02, 0.05]

    SAVE_NAME = (
        f"noise_positive_f{N_FEATURES}_s{N_GAMMA_SAMPLES*N_SINGLE_MASK_SAMPLES}"
    )

    sim_fns = {
        "exp_log_and": jit(sbdr.exp_log_and),
        "exp_log_and_delta": jit(sbdr.exp_log_and_delta),
    }


    MIs = {sim_name: [] for sim_name in sim_fns.keys()}

    verbose = False
    disable_tqdm = verbose

    for n_nonzero in tqdm(GAMMA_AVERAGE_NON_ZERO, disable=disable_tqdm):

        for k in MIs.keys():
            MIs[k].append([])

        if verbose:
            print(f"\nn_nonzero: {n_nonzero}")

        for i, noise_p in enumerate(
            tqdm(NOISE_PROBS, leave=False, disable=disable_tqdm)
        ):

            for k in MIs.keys():
                MIs[k][-1].append([])

            if verbose:
                print(f"  noise_p: {noise_p}")

            for seed in tqdm(SEEDS, leave=False, disable=disable_tqdm):

                if verbose:
                    print(f"\tseed: {seed}")

                # Generate the probabilities for the activation mask
                # seed = int(time())
                key = jax.random.key(seed)

                mask_ps, (alpha, beta) = generate_ps_beta_distribution(
                    key,
                    n_mean_active=n_nonzero,
                    n_total_features=N_FEATURES,
                    concentration=GAMMA_CONCENTRATION,
                    n_batch_size=N_GAMMA_SAMPLES,
                )

                if verbose:
                    print(f"\t  alpha: {alpha}, beta: {beta}")

                # Bernoulli-sample N_SINGLE_MASK_SAMPLES samples from the activation probabilties given by
                # N_GAMMA_SAMPLES samples from the gamma distributions
                key, _ = jax.random.split(key)
                zs_mask = jax.random.bernoulli(
                    key, mask_ps, shape=(N_SINGLE_MASK_SAMPLES, *(mask_ps.shape))
                )
                zs_mask = zs_mask.reshape((-1, N_FEATURES))

                # Add noise to the mask
                key, _ = jax.random.split(key)
                zs_mask_noised = zs_mask + jax.random.bernoulli(
                    key, p=noise_p, shape=zs_mask.shape
                )
                zs_mask_noised = np.clip(zs_mask_noised, 0, 1)

                if verbose:
                    print(f"\t  zs_mask.shape: {zs_mask.shape}")
                    print(
                        f"\t  average non-zero units: {zs_mask.sum(axis=-1).mean(axis=-1)}"
                    )

                # # Sample uniformly from the activation_range the activation value of each unit
                key, _ = jax.random.split(key)
                activation_values = jax.random.uniform(
                    key,
                    shape=zs_mask.shape,
                    minval=UNIFORM_RANGE[0],
                    maxval=UNIFORM_RANGE[1],
                )

                # Apply the mask
                zs = activation_values * zs_mask
                zs_noised = activation_values * zs_mask_noised
                # print(zs.mean(), zs.std(), zs.min(), zs.max())
                # zs = np.clip(zs, 0.001, 1)

                for sim_name, sim_fn in sim_fns.items():
                    # self-similarity
                    p_ii = sim_fn(zs, zs_mask_noised, eps=EPS_SIM)
                    # cross-similarity
                    p_ij = sim_fn(zs[:, None, ...], zs_mask_noised[None, :, ...], eps=EPS_SIM)

                    # check for nans in p_ij
                    # if np.any(np.isnan(p_ij)):
                    #     raise ValueError("p_ij has nan")

                    # Compute InfoNCE k-sample estimator
                    pmis = np.log(p_ii / (p_ij.mean(axis=-1) + 1e-6) + 1e-6)
                    # if np.any(np.isnan(pmis)):
                    #     raise ValueError("pmis has nan")

                    mi = pmis.mean(axis=-1)

                    MIs[sim_name][-1][-1].append(mi.item())

    for sim_name in sim_fns.keys():
        print(f"MIs.shape: {np.array(MIs[sim_name]).shape}")

    # create save dictionary

    data = {
        "MI": MIs,
        "n_nonzero": GAMMA_AVERAGE_NON_ZERO,
        "gamma_concentration": GAMMA_CONCENTRATION,
        "uniform_range": UNIFORM_RANGE,
        "noise_probs": NOISE_PROBS,
        "seeds": SEEDS.tolist(),
        "eps_sim": EPS_SIM,
    }

    if SAVE:
        with open(os.path.join(save_folder, SAVE_NAME + ".json"), "w") as f:
            json.dump(data, f)
