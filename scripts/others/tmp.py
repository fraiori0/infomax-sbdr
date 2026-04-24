import jax
from jax import vmap, grad, jit
import jax.numpy as np
import plotly.graph_objects as go
import sklearn as skl
from sklearn.decomposition import PCA

np.set_printoptions(precision=4, suppress=True)

SEED = 986

in_features = 9
out_features = 11


key = jax.random.PRNGKey(SEED)

def f(params, x):
    al_e = x @ params["wl_e"]
    al_i = x @ params["wl_i"]
    a = al_e - al_i

    zero_e = al_e - jax.lax.stop_gradient(al_e)
    a = np.where(a > 0, a, zero_e)

    f_val = a.mean()

    aux = {
        "a": a,
    }

    return f_val, aux

df = grad(f, argnums=0, has_aux=True)


key, subkey = jax.random.split(key, 2)
params = {
    "wl_e": jax.random.normal(key, (in_features, out_features)),
    "wl_i": jax.random.normal(subkey, (in_features, out_features)),
}
key, _ = jax.random.split(key, 2)
x = jax.random.normal(key, (in_features,))


# Compute f
f_val, aux = f(params, x)
print(f_val)

print(aux["a"].shape)
print(aux["a"])

df_val, aux = df(params, x)

for i in range(out_features):
    print(f"\na: {aux['a'][i]}")
    print(f"wl_e: {df_val['wl_e'][:, i]}")
    print(f"wl_i: {df_val['wl_i'][:, i]}")


