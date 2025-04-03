import jax
import jax.numpy as np
from jax import jit, grad, vmap
from functools import partial
import numpy as onp


def gen_decaying_weights(gamma, eps=1e-2):
    # Given gamma, compute the amount of steps after which the remaining geometric serie amount
    # to less then eps*total_discounted_sum
    # we want (1.0 - (1.0-gamma**(n_steps+1))) < eps
    total = 1.0 / (1.0 - gamma)
    n_steps_sr = int(onp.log(eps * total) / onp.log(gamma) + 1)
    ws = np.array([gamma**i for i in range(n_steps_sr)])

    return ws


sr_batch_correlate = vmap(
    partial(np.correlate, mode="valid"), in_axes=(0, None), out_axes=0
)


def gen_sr(observations, ws):
    reverse_ws = np.flip(ws)
    n_steps = ws.shape[0]

    observations = observations.reshape(observations.shape[0], -1)

    sr_obs = sr_batch_correlate(observations.T, ws).T
    sr_reverse_obs = sr_batch_correlate(observations.T, reverse_ws).T

    return (
        observations[n_steps - 1 : -n_steps + 1],
        sr_obs[n_steps - 1 :],
        sr_reverse_obs[: -n_steps + 1],
    )
