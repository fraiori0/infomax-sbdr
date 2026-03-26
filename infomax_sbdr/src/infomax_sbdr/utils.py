import os
import json
import pickle
import jax.numpy as np
import orbax.checkpoint
import jax


# convolution operator over one specified axis, useful for time convolution with arbitrary batch dimensions
def conv1d(x, w, axis: int, mode="valid"):
    return np.apply_along_axis(lambda x: np.convolve(x, w, mode=mode), axis, x)

def sigmoid_ste(x):
    """ Sigmoid activation with straight-through gradient """
    zero = x - jax.lax.stop_gradient(x)
    return zero + jax.nn.sigmoid(jax.lax.stop_gradient(x))

def threshold_softgradient(x, threshold=0.0):
    """ Threshold function with the gradient of a sigmoid"""
    zero = jax.nn.sigmoid(x-threshold) - jax.lax.stop_gradient(jax.nn.sigmoid(x-threshold))
    return zero + ((x - threshold) >= 0).astype(x.dtype)

def hard_threshold(x):
    return (x > 0).astype(x.dtype)

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))

def get_shapes(pytree):
    return jax.tree.map(lambda x: x.shape, pytree)



def print_pytree_shapes(pytree, prefix=""):
    if isinstance(pytree, dict):
        for key, value in pytree.items():
            print(f"{prefix}{key}:")
            print_pytree_shapes(value, prefix + "\t")
    elif isinstance(pytree, (list, tuple)):
        for idx, value in enumerate(pytree):
            print(f"{prefix}[{idx}]:")
            print_pytree_shapes(value, prefix + "\t")
    else:
        print(f"{prefix}{pytree.shape}")


def save_model(params, opt_state, model_path, verbose=True):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(os.path.join(model_path, "params.pkl"), "wb") as f:
        pickle.dump(params, f)

    with open(os.path.join(model_path, "opt_state.pkl"), "wb") as f:
        pickle.dump(opt_state, f)

    if verbose:
        print(f"Model saved: {model_path}")


def load_model(model_path, verbose=True):
    with open(os.path.join(model_path, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    with open(os.path.join(model_path, "opt_state.pkl"), "rb") as f:
        opt_state = pickle.load(f)

    if verbose:
        print(f"Model loaded: {model_path}")

    return params, opt_state


def make_grid_time_encoder(T_min, T_max, K, N):
    """
    Construct a multi-scale binary grid positional encoder.

    Partitions time into K geometric scales, each divided into N bins.
    The code for time t is the concatenation of K one-hot vectors,
    one per scale, giving a KN-dimensional binary vector with exactly
    K active bits.

    Args:
        T_min (float): Period of the finest scale (same units as input time).
                       Sets resolution: finest bin width = T_min / N.
        T_max (float): Period of the coarsest scale.
                       Sets unambiguous range ≈ T_max.
        K (int):       Number of scales. Controls total dimension and
                       Hamming capacity: max distance = 2K.
        N (int):       Bins per scale. Controls sparsity (p = 1/N)
                       and resolution jointly with T_min.

    Returns:
        periods (list[float]): The K period values, for inspection.
        encode  (callable):    Function t -> code where:
                                 t:    jax array, arbitrary shape (...), float32
                                 code: jax array, shape (..., K*N), float32,
                                       binary with exactly K ones.

    Example:
        # Encode frame positions within a 10-frame window
        # at target sparsity p = 1/8, dimension D = 48
        periods, encode = make_grid_time_encoder(
            T_min=1.0, T_max=10.0, K=6, N=8
        )
        code = encode(np.array([0.0, 3.5, 9.9]))  # shape (3, 48)
    """
    # --- Compute periods using standard Python math (not JAX-traced) --------
    # This runs once at construction time; the result is a fixed constant.

    if K == 1:
        periods_py = [float(T_min)]
    else:
        log_min = np.log(T_min)
        log_max = np.log(T_max)
        periods_py = [
            np.exp(log_min + (log_max - log_min) * k / (K - 1))
            for k in range(K)
        ]

    # --- Capture as JAX constants in the closure ----------------------------
    # Shape (K,) and (N,) — treated as compile-time constants by XLA.
    periods = np.array(periods_py, dtype=np.float32)   # (K,)
    bin_ids = np.arange(N, dtype=np.int32)             # (N,)

    # --- Encoder function ---------------------------------------------------
    def encode(t):
        """
        Args:
            t: jax array of shape (...), arbitrary batch dimensions.
        Returns:
            code: jax array of shape (..., K*N), float32, binary.
                  Exactly K entries are 1.0, the rest are 0.0.
        """
        t = np.asarray(t, dtype=np.float32)          # (...)

        # (..., 1) broadcast against (K,) -> (..., K)
        # Phase within each period, in [0, 1).
        # np.mod follows Python convention: result has sign of divisor,
        # so negative times are handled correctly (mod is always >= 0).
        phases = np.mod(t[..., np.newaxis], periods) / periods    # (..., K)

        # Integer bin index in {0, ..., N-1} at each scale.
        bin_idx = np.floor(phases * N).astype(np.int32)           # (..., K)

        # One-hot per scale via broadcast comparison:
        #   bin_idx[..., np.newaxis] : (..., K, 1)
        #   bin_ids                  : (N,)
        #   result                   : (..., K, N), dtype bool -> float32
        code = (bin_idx[..., np.newaxis] == bin_ids).astype(np.float32)

        # Flatten the (K, N) trailing axes into a single (K*N,) axis.
        # t.shape is always a tuple of Python ints in JAX, safe inside jit.
        return code.reshape(t.shape + (K * N,))                   # (..., K*N)

    return periods_py, encode