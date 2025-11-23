import jax
import jax.numpy as np
import flax.linen as nn
from functools import partial


class NeuralGas(nn.Module):    
    """Implementation of a Neural Gas model"""

    in_features: int
    n_units: int
    epsilon_0: float = 0.3
    epsilon_f: float = 0.01
    lambda_f: float = 0.1
    topk: float = 10

    def setup(self):

        self.lambda_0 = self.n_units * 1.0

        # the parameter is the position of the centroid associated to each unit
        self.c = self.param(
            "c",
            nn.initializers.truncated_normal(stddev=1.0, dtype=np.float32),
            (self.n_units, self.in_features),
            np.float32,
        )

    
    def __call__(self, x):

        # Compute the distance from each centroid
        d = np.sqrt(((x[..., None, :] - self.c)**2).sum(-1))

        # Compute the rank of each centroid (0 for the closest, self.n_unit-1 for farthest)
        i_sort = np.argsort(d, axis=-1)
        k = np.argsort(i_sort, axis=-1)

        # print("\n<<<<<<<<.>>>>>>>>>")
        # print(k[:5])

        # # # Activate the k closest unit
        idx_topk = i_sort[..., :self.topk]
        z = np.zeros((*x.shape[:-1], self.n_units), dtype=np.float32)
        # # k-hot encoding
        # z = np.put_along_axis(z, idx_topk, 1.0, axis=-1, inplace=False)
        # Instead, use the rank rescaled in 0-1
        z = np.put_along_axis(z, idx_topk, (1.0/np.arange(1.0, self.topk+1)), axis=-1, inplace=False)
    
        # Return also the value of the single closest centroid
        x_c = self.c[i_sort[..., 0]]

        return {
            "d": d,
            "i_sort": i_sort,
            "k": k,
            "z": z,
            "x_c": x_c
        }

    
    def param_schedule(self, t):
        # t \in [0, 1]
        eps = self.epsilon_0 * (self.epsilon_f/self.epsilon_0)**t
        lam = self.lambda_0 * (self.lambda_f/self.lambda_0)**t
        return {"eps":eps, "lam":lam}
    
    # def compute_dc(self, x, out, t):
    #     """https://en.wikipedia.org/wiki/Neural_gas#Algorithm"""

    #     # get training parameters from schedule, given the current time $t \in [0, 1]$
    #     el = self.param_schedule(t)
    #     eps = el["eps"]
    #     lam = el["lam"]

    #     # Compute updates to the parameters
    #     ks = out["k"]
    #     dc = eps * (np.exp(-ks/lam)[..., None]) * (x[..., None, :] - self.c)
        
    #     # print(ks[:3])
    #     # print(dc.shape)
    #     # exit()

    #     # Sum of all updates on all batch dimensions
    #     dc = dc.mean(axis=0)

    #     return {
    #         "c": dc
    #     }
    
    def compute_dc(self, x, out, t):
        """https://en.wikipedia.org/wiki/Neural_gas#Algorithm"""

        # get training parameters from schedule, given the current time $t \in [0, 1]$
        el = self.param_schedule(t)
        eps = el["eps"]
        lam = el["lam"]

        # Get distances
        d = out["d"]  # shape: (batch, n_units)
        
        # Compute rank of each centroid for each sample
        # ranks[i, j] = rank of centroid j for sample i (0 = closest, n_units-1 = farthest)
        k = out["k"] # ranks = np.argsort(np.argsort(d, axis=-1), axis=-1)
        
        # Compute neighborhood function based on ranks
        h = np.exp(-k / lam)  # shape: (batch, n_units)
        
        # Compute updates for all centroids
        # dc[i, j, :] = update for centroid j from sample i
        dc = eps * h[..., None] * (x[..., None, :] - self.c)  # shape: (batch, n_units, in_features)
        
        # Average over all batch dimensions
        dc = dc.mean(axis=0)  # shape: (n_units, in_features)

        return {"c": dc}
        
    
    def compute_dparams(self, x, out, t):

        dparams = {
            "params": self.compute_dc(x, out, t)
        }

        return dparams
        

    def update_params(self, params, x, out, t):
        dparams =  self.compute_dparams(x, out, t)

        params["params"]["c"] = params["params"]["c"] + dparams["params"]["c"]

        return params, dparams

    def activate_topk(self, x, k):
        # Encode inputs by activating the units corresponding to the k closest centroids
        # xs : (*batch_dims, features)
        # k : int
        # coimpute distance and take centers
        ds = self(x)
        cs = self.c
        # take the closest topk centers (note, we do topk on -ds)
        idx_topk = jax.lax.top_k(-ds, k=k)[1]
        cs_topk = cs[..., idx_topk, :]

        # return also a k-hot encoding of the centroids, of shape (*batch_dims, n_centers)
        # with the topk closest unit (for each sample) set to 1, and 0 everywhere else
        z = np.zeros((*x.shape[:-1], self.n_units), dtype=np.float32)
        z = np.put_along_axis(z, idx_topk, 1.0, axis=-1, inplace=False)

        outs = {
            "topk_c": cs_topk,
            "topk_i": idx_topk,
            "d": ds,
            "z": z,
        }
        return outs