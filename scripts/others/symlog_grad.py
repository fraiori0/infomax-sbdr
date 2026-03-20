import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn as skl
from sklearn.decomposition import PCA
import infomax_sbdr as sbdr
import jax.numpy as jnp
from jax import grad, vmap 



def f(x):
    return sbdr.symlog(x)


# plot the function and its gradient in two separate subplots
x = jnp.linspace(-1, 1, 100)
y = f(x)
grad_f = vmap(grad(f))(x)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

fig.add_trace(
    go.Scatter(x=x, y=y, mode="lines", name="f(x)"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=grad_f, mode="lines", name="grad f(x)"),
    row=2, col=1
)

fig.update_layout(height=600, width=800)
fig.show()