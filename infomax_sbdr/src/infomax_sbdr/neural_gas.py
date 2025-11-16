import jax
import jax.numpy as np
import flax.linen as nn
from functools import partial


class NeuralGas(nn.Module):    
    """Implementation of a Neural Gas model"""

    features: int
    n_units: int
    epsilon_0: float = 0.3
    epsilon_f: float = 0.01
    lambda_f: float = 0.01

    def setup(self):

        self.lambda_0 = self.n_units * 1.0

        # the parameter is the position of the centroid associated to each unit
        self.c = self.param(
            "c",
            nn.initializers.lecun_normal(),
            (self.n_units, self.features),
            np.float32,
        )

    
    def __call__(self, x):

        # Compute the distance from each centroid
        ds = ((x[..., None, :] - self.c)**2).sum(-1)

        # compute the centroids
        idx_topk = jax.lax.top_k(-ds, k=k)[1]

        cs_topk = self.c[..., idx_topk, :]
        ds_topk = np.sqrt(ds[..., idx_topk])

        # return also a aux with a k-hot encoding of the centroids, shape (*batch_dims, n_centers)
        z = np.zeros((*x.shape[:-1], self.n_units), dtype=np.float32)
        z = z.at[..., idx_topk].set(1.0)

        outs = {
            "i_topk": idx_topk,
            "d": ds_topk,
            "z": z,
        }

        return outs

    
    def param_schedule(self, t):
        # t \in [0, 1]
        eps = self.epsilon_0 (self.epsilon_f/self.epsilon_0)**t
        lam = self.lambda_0 (self.lambda_f/self.lambda_0)**t
        return {"eps":eps, "lam":lam}
    
    def update_params(self, params, x, outs, t):
        # t in [0,1]
        pass