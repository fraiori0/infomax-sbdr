import jax
import jax.numpy as np
from jax import jit, grad, vmap

np.set_printoptions(precision=4, suppress=True)


def f(x):
    return np.roll(x, 1, axis=-1)

df = jax.jacfwd(f)

x = np.linspace(0,1,10)

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


x = np.linspace(0, 1, 20*64*10).reshape((20, 64, 10))

shift_kernel = make_shift_conv_kernel(window_size=3, feature_dim=x.shape[-1]) 

x_shift = jax.lax.conv_general_dilated(
    x,
    shift_kernel,
    window_strides=(1,),
    padding="SAME",
    dimension_numbers=("NWC", "WIO", "NWC"),
)

print(x.shape)
print(x_shift.shape)

print(x[0])
print(x_shift[0])