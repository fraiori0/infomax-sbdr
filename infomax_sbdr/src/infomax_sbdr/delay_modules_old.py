import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
import numpy as onp

from typing import Sequence, Callable, Tuple, NamedTuple, Optional
from functools import partial

import infomax_sbdr.utils as ut
import infomax_sbdr.initializers as my_inits

Array = jax.Array

class Buffer(NamedTuple):
    """
    Circular buffer with optional leading batch dimensions.

    data : Array, shape [*batch, capacity, n_features]
        Stored time steps.  Batch dimensions (if any) sit in front; capacity
        and features are always the last two axes.
    ptr : Array, scalar int32
        Next write slot.  Scalar (shared across all batch elements: every
        element in a batch advances its buffer in lockstep).

    Being a NamedTuple of JAX arrays this is automatically a valid JAX pytree
    and travels through lax.scan / jit / vmap without any registration.
    """
    data: Array   # [*batch, capacity, n_features]
    ptr:  Array   # scalar int32, shared across batch elements


def buffer_init(
    capacity: int,
    n_features: int,
    batch_shape: tuple[int, ...] = (),
) -> Buffer:
    """
    Return a zero-filled Buffer of the given capacity.

    Parameters
    ----------
    capacity : int
        Number of time steps to store (= max_delay + 1).
    n_features : int
        Feature dimension (n_in for input buffers, n_hidden for state buffers).
    batch_shape : tuple of int
        Leading batch dimensions.  () for no batching.
        Typically obtained from example_input.shape[:-1].
    """
    return Buffer(
        data=np.zeros((*batch_shape, capacity, n_features)),
        ptr=np.zeros((), dtype=np.int32),
    )


def buffer_push(buf: Buffer, x: Array) -> Buffer:
    """
    Write x into the current write slot and advance the pointer.

    Parameters
    ----------
    buf : Buffer
        data shape: [*batch, capacity, n_features]
    x : Array, shape [*batch, n_features]
        Current input.  Batch shape must match buf.data.

    Returns
    -------
    Buffer with data shape unchanged; ptr incremented mod capacity.

    Cost: O(n_features) per batch element.
    `buf.data.at[..., ptr, :].set(x)` writes one slice along dim -2
    (capacity) for every batch element simultaneously.  XLA fuses this
    scatter into an in-place update inside a lax.scan.
    """
    data = buf.data.at[..., buf.ptr, :].set(x)  # dim -2 = capacity axis
    ptr  = (buf.ptr + 1) % buf.data.shape[-2]
    return Buffer(data=data, ptr=ptr)


def buffer_gather(buf: Buffer, delays: onp.ndarray | Array) -> Array:
    """
    Gather values from the buffer at arbitrary per-connection integer delays.

    Parameters
    ----------
    buf : Buffer
        data shape: [*batch, capacity, n_features]
    delays : integer array, shape [*leading, n_features]
        delays[..., j] = d  means "read feature j from d steps ago".
        d=0  →  value from the most recent buffer_push call.
        d=capacity-1  →  oldest available slot.

    Returns
    -------
    Array of shape [*batch, *leading, n_features].
    The batch dimensions of buf.data are prepended to the leading dims of
    delays (which for a standard delayed-linear layer is [n_out, n_features]).

    Implementation note
    -------------------
    `buf.data[..., idx, feat]` where idx has shape [*leading, n_features] and
    feat = arange(n_features) compiles to a single XLA Gather kernel.

    The `...` absorbs all batch dimensions of buf.data; `idx` and `feat` then
    address dimensions -2 (capacity) and -1 (features) respectively.  Because
    the advanced indices idx/feat are contiguous and trail the basic ellipsis,
    NumPy/JAX advanced-indexing rules place them at their natural position in
    the output, giving shape [*batch, *leading, n_features].

    When buf.data has no batch dims ([capacity, n_features]) this reduces to
    the original `buf.data[idx, feat]` giving shape [*leading, n_features].
    """
    capacity   = buf.data.shape[-2]
    n_features = buf.data.shape[-1]
    idx  = (buf.ptr - 1 - delays) % capacity   # [*leading, n_features]
    feat = np.arange(n_features)               # [n_features]
    return buf.data[..., idx, feat]             # [*batch, *leading, n_features]


# ─────────────────────────────────────────────────────────────────────────────
# Delay-array helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_random_delays(
    n_in: int,
    n_out: int,
    max_delay: int,
    *,
    seed: int = 0,
) -> onp.ndarray:
    """
    Sample one independent uniform integer delay in [0, max_delay] for every
    (output unit, input feature) pair.

    Returns a plain NumPy array so it can be stored as a static module field
    without being traced by JAX.

    Parameters
    ----------
    n_in, n_out : int
        Input / output dimensionality of the layer.
    max_delay : int
        Maximum delay in time steps (inclusive).
    key : jax.Array
        PRNG key.

    Returns
    -------
    onp.ndarray of shape [n_out, n_in] with dtype int32.
    """
    rng = onp.random.default_rng(seed)
    delays = rng.integers(0, max_delay + 1, size=(n_out, n_in), dtype=onp.int32)
    return delays


def make_zero_delays(n_in: int, n_out: int) -> onp.ndarray:
    """
    All delays set to 0: recovers a standard (non-delayed) linear layer.
    Useful as a sanity-check baseline or for ablations.
    """
    return onp.zeros((n_out, n_in), dtype=onp.int32)


# ─────────────────────────────────────────────────────────────────────────────
# DelayedLinear – Flax Linen Module
# ─────────────────────────────────────────────────────────────────────────────

class DelayedLinear(nn.Module):
    """
    Linear layer where each (output unit, input feature) connection reads the
    input at its own individual fixed integer delay:

        y[i] = act( Σ_j  W[i,j] · x[t - delays[i,j], j]  +  b[i] )

    Signature
    ---------
    The module follows the RNN-cell carry-first convention so it fits
    directly into a lax.scan (or nn.scan) step:

        new_buf, y = layer.apply(variables, buf, x)

    Batch dimensions
    ----------------
    __call__ supports arbitrary leading batch dimensions natively.
    x   : [*batch, n_in]
    buf : Buffer with data shape [*batch, capacity, n_in]
    y   : [*batch, features]

    Use init_buffer(example_input=x0) to create a buffer that matches the
    batch shape of your inputs.

    Parameters
    ----------
    features : int
        Number of output units (n_out).
    delays : onp.ndarray, shape [features, n_in], dtype int
        Fixed per-connection delays.  Build with make_random_delays() or
        make_zero_delays().  Stored as a static module attribute; never traced
        by JAX, never part of the parameter tree.
    activation : callable, optional
        Element-wise activation (e.g. nn.relu, jnp.tanh).  None = linear.
    use_bias : bool
        Whether to add a learnable bias.  Default True.
    kernel_init : Flax initializer
        Initialiser for W.  Default: LeCun normal.
    bias_init : Flax initializer
        Initialiser for b.  Default: zeros.
    """
    
    features:    int
    delays:      onp.ndarray            # [features, n_in], int – STATIC, never traced
    use_bias:    bool = True
    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init:   Callable = jax.nn.initializers.zeros

    # ── Static properties (pure Python/NumPy, evaluated at trace time) ────────

    @property
    def n_in(self) -> int:
        """Input dimensionality, inferred from delays.shape[1]."""
        return int(self.delays.shape[1])

    @property
    def max_delay(self) -> int:
        """Maximum delay present in the delays array."""
        return int(self.delays.max())

    @property
    def buffer_capacity(self) -> int:
        """Number of slots the circular buffer must hold (= max_delay + 1)."""
        return self.max_delay + 1

    # ── Flax setup / call ─────────────────────────────────────────────────────

    def setup(self) -> None:
        self.kernel = self.param(
            'kernel',
            self.kernel_init,
            (self.n_in, self.features),
        )
        if self.use_bias:
            self.bias = self.param(
                'bias',
                self.bias_init,
                (self.features,),
            )

    def __call__(self, buf: Buffer, x: Array) -> tuple[Buffer, Array]:
        """
        Process one time step.

        Parameters
        ----------
        buf : Buffer
            Circular buffer with data shape [*batch, capacity, n_in].
            Initialise with self.init_buffer() or
            self.init_buffer(example_input=x0) before the first step.
        x : Array, shape [*batch, n_in]
            Input for the current time step.  Batch dims must match buf.

        Returns
        -------
        new_buf : Buffer
            Updated buffer (x pushed in, pointer advanced).
        y : Array, shape [*batch, features]
            Layer output for this time step.
        """
        # 1. Push current input; delay=0 now refers to x.
        new_buf = buffer_push(buf, x)

        # 2. Gather: gathered[..., i, j] = x[t - delays[i,j], j]
        #    Shape: [*batch, features, n_in]
        gathered = buffer_gather(new_buf, self.delays)

        # 3. Weighted sum: kernel [features, n_in] broadcasts over *batch
        activation = np.sum(self.kernel.T * gathered, axis=-1)   # [*batch, features]

        if self.use_bias:
            activation = activation + self.bias                           # bias [features] broadcasts
        
        out = {
            "a": activation,
        }

        return new_buf, out

    # ── Buffer initialisation ─────────────────────────────────────────────────

    @nn.nowrap
    def init_buffer(self, example_input: Optional[Array] = None) -> Buffer:
        """
        Return a zero-filled Buffer sized for this layer.

        Decorated with @nn.nowrap so it may be called before (or outside of)
        model.init() / model.apply() – no Flax variable scope is required.

        Parameters
        ----------
        example_input : Array of shape [*batch, n_in], optional
            If provided, the buffer's data will have leading batch dimensions
            that match example_input.shape[:-1], so the buffer is immediately
            compatible with batched inputs of that shape.
            If None, the buffer has no batch dimensions (shape [capacity, n_in]).

        Returns
        -------
        Buffer with data shape [*batch, capacity, n_in] (or [capacity, n_in]).
        """
        batch_shape = example_input.shape[:-1] if example_input is not None else ()
        return buffer_init(self.buffer_capacity, self.n_in, batch_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Usage examples
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from jax import lax

    T, n_in, n_out, max_delay = 100, 8, 32, 10
    B1, B2 = 4, 3   # two batch dimensions

    k0, k_delays, k_init = jax.random.split(jax.random.PRNGKey(0), 3)

    delays = make_random_delays(n_in, n_out, max_delay, key=k_delays)
    layer  = DelayedLinear(features=n_out, delays=delays, activation=nn.relu)

    # ── Init (no batch) ───────────────────────────────────────────────────────
    buf0      = layer.init_buffer()                     # [capacity, n_in]
    variables = layer.init(k_init, buf0, jnp.zeros(n_in))

    # ── Unbatched scan ────────────────────────────────────────────────────────
    x_seq  = jax.random.normal(k0, (T, n_in))

    @jax.jit
    def run_sequence(x_seq: Array) -> Array:
        def step(buf, x):
            return layer.apply(variables, buf, x)
        _, y_seq = lax.scan(step, layer.init_buffer(), x_seq)
        return y_seq

    y_seq = run_sequence(x_seq)
    print(f"Unbatched scan:   x {x_seq.shape}  →  y {y_seq.shape}")
    # x (100, 8) → y (100, 32)

    # ── Batched scan (arbitrary leading dims, buffer carries them too) ─────────
    x_batch = jax.random.normal(k0, (B1, B2, T, n_in))
    x0_batch = x_batch[:, :, 0, :]                    # [B1, B2, n_in]

    @jax.jit
    def run_batch(x_batch: Array) -> Array:
        # x_batch: [B1, B2, T, n_in]
        # Scan over axis 2 (time); buffer carries [B1, B2, capacity, n_in]
        x_seq  = jnp.moveaxis(x_batch, -2, 0)         # [T, B1, B2, n_in]
        buf0   = layer.init_buffer(x0_batch)            # data: [B1, B2, capacity, n_in]

        def step(buf, x):                               # x: [B1, B2, n_in]
            return layer.apply(variables, buf, x)       # → (buf, y [B1, B2, n_out])

        _, y_seq = lax.scan(step, buf0, x_seq)         # y_seq: [T, B1, B2, n_out]
        return jnp.moveaxis(y_seq, 0, -2)              # [B1, B2, T, n_out]

    y_batch = run_batch(x_batch)
    print(f"Batched scan:     x {x_batch.shape}  →  y {y_batch.shape}")
    # x (4, 3, 100, 8) → y (4, 3, 100, 32)

    # ── Gradient through kernel ───────────────────────────────────────────────
    def loss_fn(params, x_seq):
        vs = {**variables, 'params': params}
        def step(buf, x):
            return layer.apply(vs, buf, x)
        _, y_seq = lax.scan(step, layer.init_buffer(), x_seq)
        return y_seq.sum()

    grad = jax.jit(jax.grad(loss_fn))(variables['params'], x_seq)
    print(f"Kernel grad:      shape {grad['kernel'].shape},  "
          f"norm {jnp.linalg.norm(grad['kernel']):.4f}")
    # Kernel grad: shape (32, 8), norm ...

    # ── Sanity check: zero delays == standard linear ──────────────────────────
    layer_std = DelayedLinear(features=n_out, delays=make_zero_delays(n_in, n_out),
                              use_bias=False)
    vars_std  = layer_std.init(k_init, layer_std.init_buffer(), jnp.zeros(n_in))
    W         = vars_std['params']['kernel']
    x_t       = jax.random.normal(k0, (n_in,))
    y_ref     = W @ x_t
    _, y_del  = layer_std.apply(vars_std, layer_std.init_buffer(), x_t)
    print(f"Zero-delay check: max |diff| = {jnp.abs(y_del - y_ref).max():.2e}")
    # should be ~0