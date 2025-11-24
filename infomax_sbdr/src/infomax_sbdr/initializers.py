import jax
import jax.numpy as np


def bernoulli_uniform(
    p : float = 0.5,
    scale: float = 1.0,
    dtype = np.float32,
):
    """Builds an initializer that returns an array
    with each element selected from {0, scale}
    according to a bernoulli with probabiliy 'p' of being equal to scale.

    Args:
        scale: optional; the upper bound of the random distribution.
        dtype: optional; the initializer's default dtype.

    Returns:
        An initializer that returns arrays whose values are uniformly distributed in
        the range ``[0, scale)``.

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.uniform(10.0)
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
    Array([[7.298188 , 8.691938 , 8.7230015],
            [2.0818567, 1.8662417, 5.5022564]], dtype=float32)
    """

    def init(key,
            shape,
            dtype = dtype,
            out_sharding = None):
        
        dtype = jax.dtypes.canonicalize_dtype(dtype)

        return np.array(scale, dtype) * jax.random.bernoulli(key, p, shape).astype(dtype)
    
    return init