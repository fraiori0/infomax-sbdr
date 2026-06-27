import os
import jax
import jax.numpy as np
from jax import grad, vmap, jit
from pprint import pprint

np.set_printoptions(precision=5, suppress=True)

SEED = 1234

eps = 1e-2

def forward(params, x):
    return x @ params["W"] + params["b"]

def L(params, x):
    x = x.reshape((-1, x.shape[-1]))
    
    y = forward(params, x)
    y_avg = y.mean(0)

    # compute infonce with linear model
    p_ii = (y*y).sum(-1) + eps
    p_avg = (y*y_avg).sum(-1) + eps
    loss_val = np.log(p_ii / p_avg).mean()

    return loss_val

# function with the gradient computed by hand, we want to check if it is correct
def grad_analytical(params, x):
    x = x.reshape((-1, x.shape[-1]))
    x_avg = x.mean(0)
    y = forward(params, x)
    y_avg = y.mean(0)
    n = (y*y).sum(-1) + eps
    d = (y*y_avg).sum(-1) + eps

    # compute the parts of the gradients
    dl_dy = (2 * y / n[..., None] - y_avg / d[..., None])
    dy_dw = x
    dy_db = 1
    # compute also for \bar{y}
    dl_dbary = - y / d[..., None]
    dbary_dw = x_avg
    dbary_db = 1

    # compute the final gradient
    dl_dw = (
        dl_dy[..., None, :] * dy_dw[..., :, None] 
        +
        dl_dbary[..., None, :] * dbary_dw[..., :, None]
    )
    dl_db = (
        dl_dy * dy_db
        +
        dl_dbary * dbary_db
    )
    return {
        "W": dl_dw.mean(0),
        "b": dl_db.mean(0),
    }

key = jax.random.key(SEED)
# init params
key, subkey = jax.random.split(key, 2)
params = {
    "W": np.abs(jax.random.normal(key, (19, 10))),
    "b": np.abs(jax.random.normal(subkey, (10,))),
}
# init samples
key, _ = jax.random.split(key, 2)
x = np.abs(jax.random.normal(key, (117, 19)))

g_an = grad_analytical(params, x)
g_auto = grad(L)(params, x)

diff = jax.tree.map(lambda x, y: x - y, g_an, g_auto)

pprint(diff["W"])
pprint(diff["b"])

# pprint(g_an["W"])
pprint(g_an["b"])

# pprint(g_auto["W"])
pprint(g_auto["b"])