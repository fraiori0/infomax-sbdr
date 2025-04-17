import jax
import jax.numpy as np
from jax import jit, grad, vmap

from binary_comparisons import *


key = jax.random.PRNGKey(123)

n_features = 50

p1 = 0.99
p2 = 0.01

n_samples = 10000


key, _ = jax.random.split(key)

ys1 = jax.random.bernoulli(key, p1, shape=(n_samples, n_features))
ys2 = jax.random.bernoulli(key, p2, shape=(n_samples, n_features))

ps1 = np.array([p1] * n_features)
ps2 = np.array([p2] * n_features)

# print(ys.shape)

wrong_exp = expected_custom_index(ps1, ps2)

# print(wrong_exp.shape)

good_exp = expected_custom_index(ys1[:, None], ys2[None, :])
print(good_exp.shape)

g_mean = good_exp.mean()

# print(g_mean.shape)

abs_diff = np.abs(g_mean - wrong_exp)

print(f"Mean abs_diff: {abs_diff.mean()} ({abs_diff.std()})")
print(f"\tMax abs_diff: {abs_diff.max()}")
print(f"\tMin abs_diff: {abs_diff.min()}")

diff = g_mean - wrong_exp
print(f"Mean diff: {diff.mean()} ({diff.std()})")
print(f"\tMax diff: {diff.max()}")
print(f"\tMin diff: {diff.min()}")
