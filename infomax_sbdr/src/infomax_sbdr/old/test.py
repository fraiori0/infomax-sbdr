import jax
import jax.numpy as np
from jax import jit, grad, vmap

from binary_comparisons import *


key = jax.random.PRNGKey(124)

n_features = 3

n_trials = 100
n_samples = 200

samples = []

key, subkey = jax.random.split(key)

ps = jax.random.uniform(key, shape=(n_trials, n_features))

ys = jax.random.bernoulli(subkey, ps[:, None], shape=(
    ps.shape[0], n_samples, ps.shape[1]))

# print(ys.shape)

wrong_exp = expected_custom_index(ps[:, None], ps[None, :])

# print(wrong_exp.shape)

good_exp = expected_custom_index(ys[:, None, :, None], ys[None, :, None, :])
# print(good_exp.shape)


g_mean = np.mean(good_exp, axis=(-2, -1))

# print(g_mean.shape)

abs_diff = np.abs(g_mean - wrong_exp)

print(f"Diff shape: {abs_diff.shape}")

print(f"Mean abs_diff: {abs_diff.mean()} ({abs_diff.std()})")
print(
    f"\tMax abs_diff: {abs_diff.max()}, at {np.unravel_index(np.argmax(abs_diff), abs_diff.shape)}")
print(f"\tMin abs_diff: {abs_diff.min()}")

diff = g_mean - wrong_exp
print(f"Mean diff: {diff.mean()} ({diff.std()})")
print(f"\tMax diff: {diff.max()}")
print(f"\tMin diff: {diff.min()}")


print(ps[34])
print(ps[31])
