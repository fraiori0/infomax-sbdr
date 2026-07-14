import os
import json
import pickle
import jax.numpy as np
import orbax.checkpoint
import jax
from functools import partial


# convolution operator over one specified axis, useful for time convolution with arbitrary batch dimensions
def conv1d(x, w, axis: int, mode="valid"):
    return np.apply_along_axis(lambda x: np.convolve(x, w, mode=mode), axis, x)

def strided_time_conv(x, weights, stride):
    """
    x: shape (*batch_dims, time, features)
    weights: shape (kernel_size,)
    stride: int
    """
    *batch_dims, time, features = x.shape

    # Collapse batch dims
    x_reshaped = x.reshape(-1, time, features)  # (B, T, C)

    # Add spatial dim structure expected by conv:
    # (N, spatial, channels)
    # For 1D conv: use (N, T, C)
    
    # Reshape to match conv_general_dilated expectations:
    # (N, C, T)
    x_conv = np.transpose(x_reshaped, (0, 2, 1))  # (B, C, T)

    # Prepare kernel for depthwise conv:
    kernel_size = weights.shape[0]

    # Shape: (out_chan, in_chan/groups, kernel_size)
    # For depthwise: out_chan = in_chan, groups = in_chan
    kernel = np.tile(weights[None, None, :], (features, 1, 1))  # (C, 1, K)

    # Perform convolution
    y = jax.lax.conv_general_dilated(
        x_conv,
        kernel,
        window_strides=(stride,),
        padding="VALID",
        dimension_numbers=("NCT", "OIT", "NCT"),
        feature_group_count=features  # depthwise
    )

    # Back to (B, windows, features)
    y = np.transpose(y, (0, 2, 1))

    # Restore original batch dims
    out_shape = (*batch_dims, y.shape[1], features)
    return y.reshape(out_shape)



def sigmoid_ste(x):
    """ Sigmoid activation with straight-through gradient """
    zero = x - jax.lax.stop_gradient(x)
    return zero + jax.nn.sigmoid(jax.lax.stop_gradient(x))

def threshold_softgradient(x):
    """ Threshold function with the gradient of a sigmoid"""
    val = jax.nn.sigmoid(x)
    zero = val - jax.lax.stop_gradient(val)
    actual_val = (x >= 0).astype(x.dtype)
    return zero + actual_val

def threshold_softpp(x, a_min=0.0, a_max=1.0):
    """ Threshold function with the gradient of a sigmoid"""
    val = jax.nn.sigmoid(x)
    zero = val - jax.lax.stop_gradient(val)
    actual_val = a_max * (x >= 0).astype(x.dtype) + a_min * (x < 0).astype(x.dtype)
    return zero + actual_val


def hard_threshold(x):
    return (x > 0).astype(x.dtype)

def threshold_ste(x):
    """ Threshold function with straight-through gradient """
    zero = x - jax.lax.stop_gradient(x)
    return zero + jax.lax.stop_gradient((x > 0).astype(x.dtype))

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))

def threshold_symlog(x):
    zero = symlog(x) - jax.lax.stop_gradient(symlog(x))
    return zero + jax.lax.stop_gradient((x > 0).astype(x.dtype))


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


@partial(jax.jit, static_argnames=("n_labels",))
def label_means(
    activations: jax.Array,   # (*batch_dims, T, F)
    labels:      jax.Array,   # (*batch_dims, T)  — integers in [0, n_labels)
    n_labels:    int,
) -> jax.Array:               # (n_labels, F)
    """Mean activation per label, averaged over all batch dims and time."""
    flat_acts = activations.reshape((-1, activations.shape[-1]))
    flat_labels = labels.reshape((-1,))

    sums   = jax.ops.segment_sum(flat_acts,                    flat_labels, n_labels)  # (n_labels, F)
    counts = jax.ops.segment_sum(np.ones(flat_labels.shape), flat_labels, n_labels)  # (n_labels,)

    return sums /(counts[:, None]+1e-6)   # (n_labels, F)


# ── 2. leave-one-out means ────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("n_labels",))
def loo_label_means(
    activations: jax.Array,   # (*batch_dims, T, F)
    labels:      jax.Array,   # (*batch_dims, T)
    n_labels:    int,
) -> jax.Array:               # (n_labels, F)
    """
    For each label k, mean of every activation whose label is NOT k.

    Uses the complement trick:
        loo_sum[k]   = total_sum   - label_sum[k]
        loo_count[k] = total_count - label_count[k]
    so the cost is identical to computing plain per-label means.
    """
    flat_acts = activations.reshape((-1, activations.shape[-1]))
    flat_labels = labels.reshape((-1,))

    sums   = jax.ops.segment_sum(flat_acts,                    flat_labels, n_labels)  # (n_labels, F)
    counts = jax.ops.segment_sum(np.ones(flat_labels.shape), flat_labels, n_labels)  # (n_labels,)

    total_sum   = sums.sum(axis=0, keepdims=True)   # (1, F)   — broadcast over labels
    total_count = counts.sum()                       # scalar

    loo_sums   = total_sum   - sums              # (n_labels, F)
    loo_counts = total_count - counts            # (n_labels,)

    return loo_sums / (loo_counts[:, None] + 1e-6)        # (n_labels, F)


# @jax.custom_vjp
# def directional_clip(x, min_val=0.0, max_val=1.0):
#     # Forward pass: Standard clip
#     return np.clip(x, min_val, max_val)

# def directional_clip_fwd(x, min_val, max_val):
#     # The forward function returns the primal output 
#     # and the residuals needed for the backward pass
#     primal_out = directional_clip(x, min_val, max_val)
#     return primal_out, (x, min_val, max_val)

# def directional_clip_bwd(res, g):
#     x, min_val, max_val = res
    
#     # In standard gradient descent, x = x - (lr * g).
#     # If g > 0, the optimizer will shrink x.
#     # If g < 0, the optimizer will grow x.

#     # Condition 1: Inside the boundary (pass gradient through)
#     in_bounds = (x >= min_val) & (x <= max_val)
    
#     # Condition 2: Above boundary AND gradient pushes x down
#     pull_down = (x > max_val) & (g > 0)
    
#     # Condition 3: Below boundary AND gradient pushes x up
#     pull_up = (x < min_val) & (g < 0)

#     # Combine conditions into a single mask
#     keep_grad_mask = in_bounds | pull_down | pull_up

#     # Apply the mask: pass upstream gradient 'g' if True, else 0.0
#     grad_x = np.where(keep_grad_mask, g, 0.0)

#     # Return gradients for all inputs (x, min_val, max_val). 
#     # We return None for the boundaries assuming they are static/non-trainable.
#     return (grad_x, None, None)

# # Bind the forward and backward passes to the custom operation
# directional_clip.defvjp(directional_clip_fwd, directional_clip_bwd)


@partial(jax.jit, static_argnames=['n', 'local_radius'])
def generate_deterministic_small_world_mask(n: int, local_radius: int) -> jax.Array:
    """
    Generates a deterministic boolean mask for small-world connectivity 
    using a Circulant Graph (Chordal Ring) approach.
    
    Args:
        n: Dimension of the square matrix (number of neurons/nodes).
        local_radius: Connects each node to `local_radius` neighbors on each side.
                      (Provides high local clustering).
        chords: A 1D jax.Array of specific long-range jump distances to add.
                (Provides short global path lengths).
        
    Returns:
        A boolean jax.numpy array of shape (n, n).
    """
    chords = np.array([
        int(n ** 0.5), # Intermediate jump
        n // 2         # Maximum distance jump
    ])

    # 1. Create periodic distance matrix for a 1D ring
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    dist = np.minimum(np.abs(i - j), n - np.abs(i - j))
    
    # 2. Local clustering mask
    # Connect to nearest neighbors, excluding self-connections (dist == 0)
    local_mask = (dist > 0) & (dist <= local_radius)
    
    # 3. Long-range shortcuts mask
    # Reshape chords to broadcast against the (n, n) distance matrix
    chords_expanded = np.asarray(chords)[:, None, None]
    
    # Check if the distance between any two nodes matches any of the chord lengths
    # np.any checks across the chord dimension (axis=0)
    long_range_mask = np.any(dist == chords_expanded, axis=0)
    
    # Combine local and long-range connections
    final_mask = local_mask | long_range_mask
    
    return final_mask