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

    SAVE_NAME = "concentration_ideal_eps1e-2"
    EPS = 1e-2

    SAVE = True

    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "resources",
        "sampled_mi",
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    N_SEEDS = 10
    SEEDS = 50 * np.arange(N_SEEDS)

    # Number of features (i.e., dimension of a single sample)
    N_FEATURES = 512

    # Number of samples drawn from the gamma distribution
    N_GAMMA_SAMPLES = 100
    # Number of binary samples drawn from each gamma distribution
    N_SINGLE_MASK_SAMPLES = 100
    # Expected number of non-zero units after Bernoulli-sampling from N_FEATURES gamma-distributed probabilities over
    GAMMA_AVERAGE_NON_ZERO = [1, 2, 3, 4, 5, 6]
    # Concentration of the gamma distribution
    GAMMA_CONCENTRATIONS = [0.1, 0.316, 1.0, 10.0, 100.0]
    # Range for the uniform sampling of the activation level, after selecting the non-zero units
    UNIFORM_RANGE = (0.9, 1.0)

    MIs = []

    verbose = False
    disable_tqdm = verbose

    for n_nonzero in tqdm(GAMMA_AVERAGE_NON_ZERO, disable=disable_tqdm):

        MIs.append([])

        if verbose:
            print(f"\nn_nonzero: {n_nonzero}")

        for i, concentration in enumerate(
            tqdm(GAMMA_CONCENTRATIONS, leave=False, disable=disable_tqdm)
        ):

            MIs[-1].append([])

            if verbose:
                print(f"  concentration: {concentration}")

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
                    concentration=concentration,
                    n_batch_size=N_GAMMA_SAMPLES,
                )

                if verbose:
                    print(f"\t  mask_ps.shape: {mask_ps.shape}")
                    print(f"\t  alpha: {alpha}, beta: {beta}")

                # Bernoulli-sample N_SINGLE_MASK_SAMPLES samples from the activation probabilties given by
                # N_GAMMA_SAMPLES samples from the gamma distributions
                keys = jax.random.split(key, N_GAMMA_SAMPLES)
                zs_mask = vmap(
                    jit(
                        partial(
                            gen_ys_k_active_weighted,
                            n_active_features=n_nonzero,
                            n_samples=N_SINGLE_MASK_SAMPLES,
                        )
                    ),
                    in_axes=(0, 0),
                    out_axes=0,
                )(
                    keys,
                    mask_ps,
                )
                zs_mask = zs_mask.reshape((-1, N_FEATURES))

                if verbose:
                    print(f"\t  zs_mask.shape: {zs_mask.shape}")
                    print(
                        f"\t  average non-zero units: {zs_mask.sum(axis=-1).mean(axis=-1)}"
                    )
                    # exit()

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
                # print(zs.mean(), zs.std(), zs.min(), zs.max())
                # zs = np.clip(zs, 0.001, 1)

                sim_fn = jit(partial(sbdr.jaccard_index, eps=EPS))
                # self-similarity
                p_ii = sim_fn(zs, zs)
                # cross-similarity
                p_ij = sim_fn(zs[:, None, ...], zs[None, :, ...])

                # check for nans in p_ij
                # if np.any(np.isnan(p_ij)):
                #     raise ValueError("p_ij has nan")

                # apply exponential if needed (depends on what we consider the critic to be)
                # p_ii = np.exp(p_ii)
                # p_ij = np.exp(p_ij)

                # Compute InfoNCE k-sample estimator
                pmis = np.log(p_ii / (p_ij.mean(axis=-1) + 1e-6) + 1e-6)
                # if np.any(np.isnan(pmis)):
                #     raise ValueError("pmis has nan")

                mi = pmis.mean(axis=-1)

                MIs[-1][-1].append(mi)

    MIs = np.array(MIs)
    print(f"MIs.shape: {MIs.shape}")

    # create save dictionary

    data = {
        "MI": MIs.tolist(),
        "n_nonzero": GAMMA_AVERAGE_NON_ZERO,
        "gamma_concentration": GAMMA_CONCENTRATIONS,
        "uniform_range": UNIFORM_RANGE,
        "seeds": SEEDS.tolist(),
    }

    if SAVE:
        with open(os.path.join(save_folder, SAVE_NAME + ".json"), "w") as f:
            json.dump(data, f)
