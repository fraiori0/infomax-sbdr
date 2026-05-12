import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable, Tuple
from functools import partial
import infomax_sbdr.binary_comparisons as bc
import infomax_sbdr.utils as ut

""" Custom pooling """

def or_pool(inputs, window_shape, strides=None, padding="VALID"):
    y = 1.0 - inputs
    y = nn.pool(y, 1.0, jax.lax.mul, window_shape, strides, padding)
    return 1.0 - y

def lag_corr_pool(
    inputs: np.ndarray,
    max_lag: int,
    window_size: int,
    stride: int = 1,
    padding: str = "VALID",
    normalize: bool = True,
) -> np.ndarray:
    """
    Windowed per-feature lag-correlation pooling.
 
    For each output position p and lag δ ∈ {0, …, max_lag}, computes:
        C_p(δ)[j] = Σ_{k=0}^{W-1-δ}  x_{p·S+k}[j] · x_{p·S+k+δ}[j]
 
    For binary x, this counts how many times feature j fires at both
    step k and step k+δ within the window — a per-feature temporal
    co-occurrence count.  Output dimension is (max_lag+1)·d instead
    of d², making this much cheaper than autocorr_pool for large d.
 
    Sparsity: for k-sparse x, at most k entries per (lag, time-step)
    are non-zero, giving a naturally sparse (max_lag+1)·d output.
 
    Args:
        inputs:      float32 array of shape (batch, T, d).
        max_lag:     Δ — maximum lag (inclusive). Produces Δ+1 lag channels.
        window_size: W — number of steps per window.
        stride:      S — step between consecutive window positions.
        padding:     "VALID" or "SAME".
        normalize:   If True, divide by (W − δ) per lag for unbiased
                     estimation (corrects for fewer valid pairs at large δ).
 
    Returns:
        float32 array of shape (batch, T_out, max_lag+1, d).
        Typically flattened to (batch, T_out, (max_lag+1)*d) before
        passing to subsequent convolutional or linear layers.
    """
    B, T, d = inputs.shape
 
    # Step 1: build all shifted copies of the input.
    # Pad max_lag zeros at the tail so that x_{t+δ} is defined for all t < T.
    inputs_padded = np.pad(
        inputs, ((0, 0), (0, max_lag), (0, 0))
    )   # (B, T + max_lag, d)
 
    # Stack δ-shifted copies:  shifted[:, :, δ, :] = inputs_padded[:, δ:δ+T, :]
    shifted = np.stack(
        [inputs_padded[:, delta : delta + T, :] for delta in range(max_lag + 1)],
        axis=2,
    )   # (B, T, max_lag+1, d)
 
    # Step 2: element-wise product of original with each shifted copy.
    # For binary inputs this is the AND operation.
    lag_products = inputs[:, :, None, :] * shifted   # (B, T, max_lag+1, d)
 
    # Step 3: flatten lag+feature dims for reduce_window.
    lp_flat = lag_products.reshape(B, T, (max_lag + 1) * d)   # (B, T, (L+1)d)
 
    # Step 4: sum-reduce over each window.
    summed = jax.lax.reduce_window(
        operand=lp_flat,
        init_value=np.zeros((), dtype=lp_flat.dtype),
        computation=jax.lax.add,
        window_dimensions=(1, window_size, 1),
        window_strides=(1, stride, 1),
        padding=padding,
    )   # (B, T_out, (L+1)d)
 
    T_out = summed.shape[1]
    result = summed.reshape(B, T_out, max_lag + 1, d)
 
    # Step 5: per-lag normalisation.
    # Number of valid (non-zero-padded) pairs in window W at lag δ = W − δ.
    if normalize:
        norm = np.array(
            [max(1, window_size - delta) for delta in range(max_lag + 1)],
            dtype=result.dtype,
        )   # (max_lag+1,)
        result = result / norm[None, None, :, None]   # broadcast over B, T_out, d
 
    return result   # (B, T_out, max_lag+1, d)


""" Index-Shift convolution kernels """

def make_shift_conv_kernel(window_size: int, feature_dim: int):
    """
    Returns kernel of shape (W, F, F)
    where each slice is a circulant shift matrix.
    """
    W, F = window_size, feature_dim

    # shifts: (W,)
    shifts = np.arange(W - 1, -1, -1)

    # base indices: (F,)
    idx = np.arange(F)

    # build (W, F, F)
    # kernel[i, out_f, in_f] = 1 if in_f == (out_f - shift) % F
    in_idx = (idx[None, :, None] - shifts[:, None, None]) % F
    out_idx = idx[None, None, :]

    kernel = (in_idx == out_idx).astype(np.float32)

    return kernel


""" Convolutional modules """

class ConvShiftLayer(nn.Module):
    """
    Module with a single dense encoder and a circulant shift convolution layer,
        similary to High-Dimensional Computing (HDC).
    We shift each element in the convolution window depending on the index,
        and sum over the window.
    """
    features: int
    conv_size: int
    conv_stride: int
    padding: str = "SAME"

    def setup(self):
        self.layer = nn.Dense(
            features=self.features,
            kernel_init=jax.nn.initializers.lecun_normal(),
        )

        self.shift_kernel = make_shift_conv_kernel(
            window_size=self.conv_size,
            feature_dim=self.features,
        )

    def __call__(self, x):
        # assume input x is of shape (*batch_dims, time, features)
        batch_shape = x.shape[:-2]

        # encode
        x = self.layer(x)

        # compute bounded pre-activation (tanh)
        a = jax.nn.tanh(x)
        # and binary activation
        z = (a > 0).astype(np.float32)

        # Convolve with Shift Kernel, for next layer
        # This represent a binding operations over vectors in a conv window
        z = z.reshape((-1, *z.shape[-2:]))
        z_conv = jax.lax.conv_general_dilated(
            z.reshape((-1, *z.shape[-2:])),
            self.shift_kernel,
            window_strides=(self.conv_stride,),
            padding=self.padding,
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        z_conv = z_conv.reshape((*batch_shape, *z_conv.shape[-2:]))

        return {
            "a": a,
            "z": z,
            "z_conv": z_conv,
        }

class ConvShift(nn.Module):

    features: Sequence[int]
    conv_sizes: Sequence[Tuple[int]]
    conv_strides: Sequence[Tuple[int, int]]
    padding: str = "SAME"

    def setup(self):
        self.layers = [
            ConvShiftLayer(
                features=self.features[i],
                conv_size=self.conv_sizes[i],
                conv_stride=self.conv_strides[i],
                padding=self.padding
            )
            for i in range(len(self.features))
        ]

    def __call__(self, x):
        
        # return all intermediate outputs
        outs = []

        for layer in self.layers:
            out = layer(x)
            outs.append(out)
            # set input to next layer
            x = jax.lax.stop_gradient(out["z_conv"])

        return outs


# standard Temporal Convolutional Network
class TemporalConvLayer(nn.Module):
    """Causal temporal convolutional layer.
 
    Attributes:
        features:      Number of output channels (kernel features).
        kernel_size:   Length of the 1-D convolutional kernel (must be ≥ 1).
        stride:        Step size along the time axis (default 1).
        use_bias:      Whether to add a learnable bias term.
        kernel_init:   Initialiser for the convolutional kernel weights.
        bias_init:     Initialiser for the bias vector.
    """
 
    features: int
    kernel_size: int
    stride: int = 1
    use_bias: bool = True
 
    # ------------------------------------------------------------------
    # setup — create sub-modules and validate hyper-parameters
    # ------------------------------------------------------------------
 
    def setup(self) -> None:
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be ≥ 1, got {self.kernel_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {self.stride}")
 
        # We manage causal padding ourselves, so the underlying Conv uses
        # 'VALID' (no implicit padding).
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding="VALID",
            use_bias=self.use_bias,
        )
 
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal temporal convolution.
 
        Args:
            x: Float array of shape ``(*batch_dims, time, features)``.
               Any number of leading batch dimensions is supported.
 
        Returns:
            Float array of shape ``(*batch_dims, time_out, self.features)``
            where ``time_out = ⌊(time − 1) / stride⌋ + 1``.
        """

        th = 0.3

        # 1. Causal padding
        # Prepend (kernel_size − 1) zero frames so each output step only
        # sees the present and past inputs, never future ones.
        #
        # np.pad expects a sequence of (before, after) pairs, one per axis.
        # Layout: [...batch axes..., time axis, feature axis]
        pad_size = self.kernel_size - 1
        if pad_size > 0:
            pad_width = [(0, 0)] * (x.ndim - 2) + [(pad_size, 0), (0, 0)]
            x = np.pad(x, pad_width)
 
        # 2. Convolution (VALID — no further padding)
        pre_activation = self.conv(x)
        z = jax.nn.sigmoid(pre_activation)
        y = ut.threshold_softgradient(pre_activation)

        outs = {
            "z": z,
            "y": y,
        }
        return outs


class TemporalConvPoolLayer(nn.Module):
    """Causal temporal convolutional layer.
 
    Attributes:
        features:      Number of output channels (kernel features).
        kernel_size:   Length of the 1-D convolutional kernel (must be ≥ 1).
        kernel_stride: Step size along the time axis (default 1).
        pool_size:     Length of the 1-D pooling kernel (must be ≥ 1).
        pool_stride:   Step size along the time axis (default 1).
        use_bias:      Whether to add a learnable bias term.
        kernel_init:   Initialiser for the convolutional kernel weights.
        bias_init:     Initialiser for the bias vector.
    """
 
    features: int
    kernel_size: int
    kernel_stride: int
    pool_size: int
    pool_stride: int
    use_bias: bool = True
 
    # ------------------------------------------------------------------
    # setup — create sub-modules and validate hyper-parameters
    # ------------------------------------------------------------------
 
    def setup(self) -> None:
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be ≥ 1, got {self.kernel_size}")
        if self.kernel_stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {self.kernel_stride}")
 
        # We manage causal padding ourselves, so the underlying Conv uses
        # 'VALID' (no implicit padding).
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.kernel_stride,),
            padding="VALID",
            use_bias=self.use_bias,
        )
 
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal temporal convolution.
 
        Args:
            x: Float array of shape ``(*batch_dims, time, features)``.
               Any number of leading batch dimensions is supported.
 
        Returns:
            Float array of shape ``(*batch_dims, time_out, self.features)``
            where ``time_out = ⌊(time − 1) / stride⌋ + 1``.
        """

        th = 0.3

        # 1. Causal padding
        # Prepend (kernel_size − 1) zero frames so each output step only
        # sees the present and past inputs, never future ones.
        #
        # np.pad expects a sequence of (before, after) pairs, one per axis.
        # Layout: [...batch axes..., time axis, feature axis]
        pad_size = self.kernel_size - 1
        if pad_size > 0:
            pad_width = [(0, 0)] * (x.ndim - 2) + [(pad_size, 0), (0, 0)]
            x = np.pad(x, pad_width)
 
        # 2. Convolution (VALID — no further padding)
        pre_activation = self.conv(x)
        z = jax.nn.sigmoid(pre_activation)
        y = ut.threshold_softgradient(pre_activation)

        # Aggregate temporally using max_pool or avg_pool
        p = nn.max_pool(
            z,
            window_shape=(self.pool_size,),
            strides=(self.pool_stride,),
            padding="SAME",
        )
        # p = z

        # # Aggregate temporally using lagged correlation
        # p = lag_corr_pool(
        #     z.reshape((-1, *z.shape[-2:])),
        #     max_lag=3,
        #     window_size=self.pool_size,
        #     stride=self.pool_stride,
        #     padding="VALID",
        # )
        # # flatten last two dimensions (lag, features)
        # p = p.reshape((*z.shape[:-2], p.shape[-3], -1))

        outs = {
            "z": z,
            "y": y,
            "p": p,
        }
        return outs


class TCN(nn.Module):
    """Temporal Convolutional Network (TCN) with multiple layers.
    """
 
    features: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int] = 1
    use_bias: bool = True
 
    def setup(self) -> None:
        if len(self.features) != len(self.kernel_sizes):
            raise ValueError("features and kernel_sizes must have the same length")
        if isinstance(self.strides, int):
            self.strides = [self.strides] * len(self.features)
        elif len(self.strides) != len(self.features):
            raise ValueError("strides must be an int or have the same length as features")
 
        self.layers = [
            TemporalConvLayer(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                use_bias=self.use_bias,
            )
            for i in range(len(self.features))
        ]

    def __call__(self, x):
        th = 0.3

        # return all intermediate outputs
        outs = []
        x_in = x
        for layer in self.layers:
            out = layer(x_in)
            outs.append(out)
            # input to next layer
            x_in = jax.lax.stop_gradient(out["y"])
            

        return outs


class TCNPoolClassifier(nn.Module):
    """Temporal Convolutional Network (TCN) with multiple layers and pooling after convolutions.
    And a final linear layer for classification using outer product of 
    sparse binary activations from the last two layers.
    """
 
    features: Sequence[int]
    kernel_sizes: Sequence[int]
    kernel_strides: Sequence[int]
    pool_sizes: Sequence[int]
    pool_strides: Sequence[int]
    class_features: int
    use_bias: bool = True
    stop_grad: bool = True
 
    def setup(self) -> None:
        if len(self.features) != len(self.kernel_sizes):
            raise ValueError("features and kernel_sizes must have the same length")
        if isinstance(self.kernel_strides, int):
            self.kernel_strides = [self.strides] * len(self.features)
        elif len(self.kernel_strides) != len(self.features):
            raise ValueError("strides must be an int or have the same length as features")
 
        self.layers = [
            TemporalConvPoolLayer(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                kernel_stride=self.kernel_strides[i],
                pool_size=self.pool_sizes[i],
                pool_stride=self.pool_strides[i],
                use_bias=self.use_bias,
            )
            for i in range(len(self.features))
        ]

        # input to the dense module is the flattened
        # outer product of the sparse representation in the two last layers
        self.classifier = nn.Dense(self.class_features)

    def __call__(self, x):
        th = 0.3

        # # # Forward psss
        # return all intermediate outputs
        outs = []
        x_in = x
        for layer in self.layers:

            out = layer(x_in)
            outs.append(out)
            
            # Input to next layer
            x_in = out["p"]
            if self.stop_grad:
                x_in = jax.lax.stop_gradient(x_in)
                
        
        # # # Classifier
        # # Compute outer product of the last two last layers' encodings
        # idx_h = max(0, len(self.layers)-2)
        # idx_g = len(self.layers)-1
        # # We use p so that it has the same time dimension than y at the next layer
        # h = outs[idx_h]["p"]
        # g = outs[idx_g]["z"]
        # # # Should we stop the gradient here?
        # # h = jax.lax.stop_gradient(h)
        # # g = jax.lax.stop_gradient(g)
        # # Compute outer product on feature dimension
        # hg = h[..., :, None] * g[..., None, :]
        # # Flatten last two dimensions
        # hg = hg.reshape((*hg.shape[:-2], -1))
        # Compute linear layer
        hg = outs[-1]["p"]
        logit = self.classifier(hg)

        aux = {
            "logit": logit,
        }

        return outs, aux
    

class TCNPoolClassifierMulti(nn.Module):
    """Temporal Convolutional Network (TCN) with multiple layers and pooling after convolutions.
    And a final linear layer for classification using outer product of 
    sparse binary activations from the last two layers.
    """
 
    features: Sequence[int]
    kernel_sizes: Sequence[int]
    kernel_strides: Sequence[int]
    pool_sizes: Sequence[int]
    pool_strides: Sequence[int]
    class_features: int
    use_bias: bool = True
    stop_grad: bool = False
 
    def setup(self) -> None:
        if len(self.features) != len(self.kernel_sizes):
            raise ValueError("features and kernel_sizes must have the same length")
        if isinstance(self.kernel_strides, int):
            self.kernel_strides = [self.strides] * len(self.features)
        elif len(self.kernel_strides) != len(self.features):
            raise ValueError("strides must be an int or have the same length as features")
 
        self.layers = [
            TemporalConvPoolLayer(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                kernel_stride=self.kernel_strides[i],
                pool_size=self.pool_sizes[i],
                pool_stride=self.pool_strides[i],
                use_bias=self.use_bias,
            )
            for i in range(len(self.features))
        ]

        # One classifier per conv layer
        self.classifiers = [
            nn.Dense(self.class_features)
            for _ in range(len(self.layers))
        ]

    def __call__(self, x):
        th = 0.3

        # # # Forward psss
        # return all intermediate outputs
        outs = []
        auxs = []
        x_in = x
        for l_idx in range(len(self.layers)):

            out = self.layers[l_idx](x_in)
            logit = self.classifiers[l_idx](out["p"])
            aux = {
                "logit": logit,
            }

            outs.append(out)
            auxs.append(aux)

            
            # Input to next layer
            x_in = out["p"]
            if self.stop_grad:
                x_in = jax.lax.stop_gradient(x_in)

        return outs, auxs