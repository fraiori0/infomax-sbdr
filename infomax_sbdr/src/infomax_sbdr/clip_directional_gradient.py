import jax
import jax.numpy as np
from functools import partial

@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def directional_clip(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)

def directional_clip_fwd(x, lo, hi):
    return np.clip(x, lo, hi), x  # residual: the pre-clip x

def directional_clip_bwd(lo, hi, x, g):
    at_lo = x <= lo
    at_hi = x >= hi
    # kill g>0 at lower bound (would push down/further below),
    # kill g<0 at upper bound (would push up/further above)
    # Original (works with gradient descent)
    mask = np.where(at_lo, g < 0, np.where(at_hi, g > 0, True))
    # # Inverted (works with gradient ascent)
    # mask = np.where(at_lo, g > 0, np.where(at_hi, g < 0, True))

    # # rescale also the gradient, so it's lower at the boundaries and max at 0.5
    sig_x = jax.nn.sigmoid(np.e * (x-0.5))
    g_scale = 4*sig_x*(1-sig_x)
    g = g * g_scale

    return (g * mask.astype(g.dtype),)

directional_clip.defvjp(directional_clip_fwd, directional_clip_bwd)



@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def directional_clip_rev(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)

def directional_clip_rev_fwd(x, lo, hi):
    return np.clip(x, lo, hi), x  # residual: the pre-clip x

def directional_clip_rev_bwd(lo, hi, x, g):
    at_lo = x <= lo
    at_hi = x >= hi
    # kill g>0 at lower bound (would push down/further below),
    # kill g<0 at upper bound (would push up/further above)
    # # Inverted (works with gradient ascent)
    mask = np.where(at_lo, g > 0, np.where(at_hi, g < 0, True))

    # rescale also the gradient, so it's lower at the boundaries and max at 0.5
    sig_x = jax.nn.sigmoid(np.e * (x-0.5))
    g_scale = 4*sig_x*(1-sig_x)
    g = g * g_scale

    return (g * mask.astype(g.dtype),)

directional_clip_rev.defvjp(directional_clip_rev_fwd, directional_clip_rev_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def directional_grad_rev(x, lo=0.0, hi=1.0):
    return x # np.clip(x, lo, hi)

def directional_grad_rev_fwd(x, lo, hi):
    return x, x  # residual: the pre-clip x

def directional_grad_rev_bwd(lo, hi, x, g):
    at_lo = x <= lo
    at_hi = x >= hi
    # kill g>0 at lower bound (would push down/further below),
    # kill g<0 at upper bound (would push up/further above)
    # # Inverted (works with gradient ascent)
    mask = np.where(at_lo, g > 0, np.where(at_hi, g < 0, True))
    return (g * mask.astype(g.dtype),)

directional_grad_rev.defvjp(directional_grad_rev_fwd, directional_grad_rev_bwd)