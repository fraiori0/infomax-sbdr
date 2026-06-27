import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
import numpy as onp
from jax.nn import initializers

from typing import Sequence, Callable, Tuple, NamedTuple, Optional
from functools import partial

import infomax_sbdr.utils as ut
import infomax_sbdr.initializers as my_inits

Array = jax.Array

# ─────────────────────────────────────────────────────────────────────────────
# Circular buffer
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Circular buffer
# ─────────────────────────────────────────────────────────────────────────────

class Buffer(NamedTuple):
    """
    Circular buffer with optional leading batch dimensions.

    data : Array, shape [*batch, capacity, n_features]
        Stored time steps.  Batch dims are in front; capacity and features
        are always the last two axes.
    ptr : Array, scalar int32
        Next write slot.  Scalar (shared across the batch: all elements
        advance their buffer in lockstep; only data differs per element).

    NamedTuple of JAX arrays → valid JAX pytree, travels through
    jax.lax.scan / jit / vmap without any registration.
    """
    data: Array   # [*batch, capacity, n_features]
    ptr:  Array   # scalar int32


def buffer_init(
    capacity: int,
    n_features: int,
    batch_shape: tuple[int, ...] = (),
) -> Buffer:
    """
    Return a zero-filled Buffer.

    Parameters
    ----------
    capacity : int
        Number of time steps retained (= max_delay + 1).
    n_features : int
        Feature width of each stored vector.
    batch_shape : tuple of int
        Leading batch dimensions.  () means no batch.
        Typically obtained from example_input.shape[:-1].
    """
    return Buffer(
        data=np.zeros((*batch_shape, capacity, n_features)),
        ptr=np.zeros((), dtype=np.int32),
    )


def buffer_push(buf: Buffer, x: Array) -> Buffer:
    """
    Write x into the current slot and advance the pointer.

    x   : [*batch, n_features] – must match buf.data's batch/feature shape.

    `buf.data.at[..., ptr, :].set(x)` writes along axis -2 (capacity) for
    every batch element simultaneously.  XLA fuses this into an in-place
    update inside jax.lax.scan.
    """
    data = buf.data.at[..., buf.ptr, :].set(x)
    ptr  = (buf.ptr + 1) % buf.data.shape[-2]
    return Buffer(data=data, ptr=ptr)


def buffer_gather(buf: Buffer, delays: Array) -> Array:
    """
    Gather values from the buffer at per-connection integer delays.

    Parameters
    ----------
    buf    : Buffer with data of shape [*batch, capacity, n_features].
    delays : integer array, shape [*leading, n_features].
             delays[..., j] = d  →  feature j, d steps ago (d=0: most recent).

    Returns
    -------
    Array of shape [*batch, *leading, n_features].

    The two-array index buf.data[..., idx, feat] compiles to a single XLA
    Gather.  The `...` absorbs all batch dims; idx and feat address dims -2
    and -1.  Because they are contiguous advanced indices trailing a basic
    ellipsis, NumPy/JAX places them at their natural position in the output.
    """
    capacity   = buf.data.shape[-2]
    n_features = buf.data.shape[-1]
    idx  = (buf.ptr - 1 - delays) % capacity   # [*leading, n_features]
    feat = np.arange(n_features)               # [n_features]
    return buf.data[..., idx, feat]             # [*batch, *leading, n_features]


# ─────────────────────────────────────────────────────────────────────────────
# Delay initializer
# ─────────────────────────────────────────────────────────────────────────────

def uniform_int_init(low: int, high: int) -> Callable:
    """
    Delay initializer factory following Flax's initializer convention.

    Returns a function with signature (key, shape, dtype) → Array that
    draws uniform integers in [low, high) – suitable for the delay_init
    field of DelayedLinear.

    Example
    -------
    layer = DelayedLinear(..., delay_init=uniform_int_init(0, max_delay + 1))
    """
    def init(key: Array, shape: tuple[int, ...], dtype=np.int32) -> Array:
        return jax.random.randint(key, shape, minval=low, maxval=high, dtype=dtype)
    return init


# ─────────────────────────────────────────────────────────────────────────────
# DelayedLinear – Flax Linen Module
# ─────────────────────────────────────────────────────────────────────────────

class DelayedLinear(nn.Module):
    """
    Linear layer where each (output unit, input feature) connection reads the
    input at its own fixed integer delay:

        y[i] = act( Σ_j  W[i,j] · x[t - delays[i,j], j]  +  b[i] )

    Delays are generated internally during model.init() using Flax's RNG
    system and stored in a dedicated 'delays' variable collection, separate
    from learnable parameters.

    Initialization
    --------------
    model.init requires two RNG keys:

        variables = layer.init({'params': key_w, 'delays': key_d}, buf0, x0)

    The 'delays' key is used once to sample the integer delay array; it is
    never needed again in model.apply().

    Signature
    ---------
    Follows the RNN-cell carry-first convention for jax.lax.scan compatibility:

        new_buf, y = layer.apply(variables, buf, x)

    Parameters
    ----------
    features : int
        Number of output units (n_out).
    n_in : int
        Number of input features.
    max_delay : int
        Maximum delay in time steps (inclusive upper bound for uniform init).
    use_bias : bool
        Whether to add a learnable bias.  Default True.
    kernel_init : Flax initializer
        Initializer for W.  Default: LeCun normal.
    bias_init : Flax initializer
        Initializer for b.  Default: zeros.
    delay_init : callable, optional
        Initializer for the delay array; signature (key, shape, dtype) → Array.
        Default: uniform integers in [0, max_delay] via uniform_int_init.
        Pass a custom function for structured or sparse delay patterns.
    """

    features:    int
    n_in:        int
    max_delay:   int
    use_bias:    bool = True
    kernel_init: Callable = initializers.lecun_normal()
    bias_init:   Callable = initializers.zeros
    delay_init:  Optional[Callable] = None  # None → uniform_int_init(0, max_delay + 1)

    @property
    def buffer_capacity(self) -> int:
        """Number of buffer slots required (= max_delay + 1)."""
        return self.max_delay + 1

    # ── Flax setup / call ─────────────────────────────────────────────────────

    def setup(self) -> None:
        # ── Delays ───────────────────────────────────────────────────────────
        # Stored in the 'delays' collection (separate from 'params').
        # The lambda body is evaluated only during model.init(), when
        # 'delays'/'kernel' does not yet exist.  During model.apply() the
        # existing value is returned directly and self.make_rng is never called.
        _delay_init = self.delay_init or uniform_int_init(0, self.max_delay + 1)
        self._delays = self.variable(
            'delays', 'kernel',
            lambda: _delay_init(
                self.make_rng('delays'),
                (self.features, self.n_in),
                np.int32,
            ),
        )

        # ── Learnable parameters ──────────────────────────────────────────────
        self.kernel = self.param(
            'kernel', self.kernel_init, (self.features, self.n_in),
        )
        if self.use_bias:
            self.bias = self.param(
                'bias', self.bias_init, (self.features,),
            )

    def __call__(self, buf: Buffer, x: Array) -> tuple[Buffer, Array]:
        """
        Process one time step.

        Parameters
        ----------
        buf : Buffer
            Circular buffer; data shape [*batch, capacity, n_in].
            Obtain the initial carry with self.init_buffer() or
            self.init_buffer(example_input=x0).
        x : Array, shape [*batch, n_in]
            Input for the current step; batch dims must match buf.

        Returns
        -------
        new_buf : Buffer  – updated carry (x pushed in, pointer advanced).
        y       : Array, shape [*batch, features]  – layer output.
        """
        new_buf = buffer_push(buf, x)

        # self._delays.value : int32 Array [features, n_in]
        # gathered           : [*batch, features, n_in]
        gathered = buffer_gather(new_buf, self._delays.value)

        # kernel [features, n_in] broadcasts over *batch
        y = np.sum(self.kernel * gathered, axis=-1)   # [*batch, features]

        if self.use_bias:
            y = y + self.bias

        return new_buf, y

    # ── Sequence scan ─────────────────────────────────────────────────────────

    def scan(
        self,
        x_seq: Array,
        buf:   Optional[Buffer] = None,
    ) -> tuple[Buffer, Array]:
        """
        Apply this layer over a full time sequence using jax.lax.scan.

        Equivalent to calling __call__ step-by-step in a loop, but compiled
        into a single XLA while-loop: O(1) intermediate memory in T and
        correct gradient flow via the scan's reverse-mode autodiff.

        Parameters
        ----------
        x_seq : Array, shape [*batch, T, n_in]
            Input sequence.  Time is expected on axis -2 (standard
            [*batch, T, features] layout).
        buf : Buffer, optional
            Initial circular buffer.  When None (default), a zero buffer is
            created automatically; its batch dimensions are inferred from
            x_seq.shape[:-2], so no example input is needed.
            Pass an explicit Buffer to continue from a previous state, e.g.
            for chunked / streaming inference over long sequences.

        Returns
        -------
        final_buf : Buffer
            State after the last time step.  Feed it back as `buf` on the
            next call to process the sequence in chunks.
        y_seq : Array, shape [*batch, T, features]
            Layer output at every time step, same axis layout as x_seq.

        Usage
        -----
        # Full sequence, auto-init:
        _, y = layer.apply(variables, x_seq, method=layer.scan)

        # Chunked (streaming), explicit carry:
        buf = layer.init_buffer(x_seq[..., 0, :])
        buf, y1 = layer.apply(variables, x_seq[:, :T//2], buf, method=layer.scan)
        buf, y2 = layer.apply(variables, x_seq[:, T//2:], buf, method=layer.scan)
        """
        if buf is None:
            # x_seq.shape is always a concrete tuple in JAX (shapes are static)
            buf = buffer_init(self.buffer_capacity, self.n_in, x_seq.shape[:-2])

        # jax.lax.scan iterates over axis 0; move the time axis there.
        x_scan = np.moveaxis(x_seq, -2, 0)      # [T, *batch, n_in]

        def step(carry: Buffer, x: Array) -> tuple[Buffer, Array]:
            return self(carry, x)                  # __call__(buf, x) → (new_buf, y)

        final_buf, y_scan = jax.lax.scan(step, buf, x_scan)  # y_scan: [T, *batch, features]
        return final_buf, np.moveaxis(y_scan, 0, -2)    # [*batch, T, features]

    # ── Buffer initialisation ─────────────────────────────────────────────────

    @nn.nowrap
    def init_buffer(self, example_input: Optional[Array] = None) -> Buffer:
        """
        Return a zero-filled Buffer sized for this layer.

        @nn.nowrap: callable before (and outside of) model.init() / apply().
        No variable scope is needed; only the static module fields are used.

        Parameters
        ----------
        example_input : Array, shape [*batch, n_in], optional
            If given, the buffer's data will carry the same leading batch
            dimensions.  If None, the buffer has no batch dimensions.
        """
        batch_shape = example_input.shape[:-1] if example_input is not None else ()
        return buffer_init(self.buffer_capacity, self.n_in, batch_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Usage examples
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    T, n_in, n_out, max_delay = 100, 8, 32, 10
    B1, B2 = 4, 3

    k_params, k_delays, k_data = jax.random.split(jax.random.PRNGKey(0), 3)

    layer = DelayedLinear(features=n_out, n_in=n_in, max_delay=max_delay,)

    # ── Init ──────────────────────────────────────────────────────────────────
    variables = layer.init(
        {'params': k_params, 'delays': k_delays},
        layer.init_buffer(), np.zeros(n_in),
    )

    # ── 1. scan: unbatched, auto-init buffer ──────────────────────────────────
    x_seq = jax.random.normal(k_data, (T, n_in))

    final_buf, y_seq = jax.jit(layer.apply, static_argnames='method')(
        variables, x_seq, method=layer.scan,
    )
    print(f"Unbatched scan:  x {x_seq.shape}  →  y {y_seq.shape}")
    # x (100, 8) → y (100, 32)

    # ── 2. scan: batched, auto-init buffer (arbitrary leading dims) ───────────
    x_batch = jax.random.normal(k_data, (B1, B2, T, n_in))

    _, y_batch = jax.jit(layer.apply, static_argnames='method')(
        variables, x_batch, method=layer.scan,
    )
    print(f"Batched scan:    x {x_batch.shape}  →  y {y_batch.shape}")
    # x (4, 3, 100, 8) → y (4, 3, 100, 32)

    # ── 3. scan: chunked / streaming (explicit carry) ─────────────────────────
    # split a sequence into two halves and process them consecutively,
    # threading the buffer carry between calls
    x_a, x_b = x_seq[:T//2], x_seq[T//2:]

    buf_init = layer.init_buffer()
    buf_mid,  y_a = jax.jit(layer.apply, static_argnames='method')(variables, x_a, buf_init, method=layer.scan)
    buf_end,  y_b = jax.jit(layer.apply, static_argnames='method')(variables, x_b, buf_mid,  method=layer.scan)
    y_chunked = np.concatenate([y_a, y_b], axis=0)

    # Should be identical to scanning the full sequence at once
    _, y_full = jax.jit(layer.apply, static_argnames='method')(variables, x_seq, buf_init, method=layer.scan)
    print(f"Chunked == full: max |diff| = {np.abs(y_chunked - y_full).max():.2e}")
    # should be ~0

    # ── 4. scan == step-by-step: equivalence check ────────────────────────────
    from jax import lax
    def step_by_step(x_seq):
        def step(buf, x):
            return layer.apply(variables, buf, x)
        _, y = jax.lax.scan(step, layer.init_buffer(), x_seq)
        return y

    y_steps  = jax.jit(step_by_step)(x_seq)
    _, y_scan = jax.jit(layer.apply, static_argnames='method')(variables, x_seq, method=layer.scan)
    print(f"scan == step-by-step: max |diff| = {np.abs(y_scan - y_steps).max():.2e}")
    # should be ~0

    # ── 5. gradient through scan ──────────────────────────────────────────────
    def loss(params, x_seq):
        vs = {**variables, 'params': params}
        _, y_seq = layer.apply(vs, x_seq, method=layer.scan)
        return y_seq.sum()

    grad = jax.jit(jax.grad(loss))(variables['params'], x_seq)
    print(f"Kernel grad:     shape {grad['kernel'].shape}, "
          f"norm {np.linalg.norm(grad['kernel']):.4f}")