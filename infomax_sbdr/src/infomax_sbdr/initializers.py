import jax
import jax.numpy as np


def bernoulli_uniform(
    p : float = 0.5,
    scale: float = 1.0,
    dtype = np.float32,
):
    """Builds an initializer that returns an array
    with each element selected from the set {0, scale}
    using to a bernoulli with probabiliy 'p' of being equal to scale.

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

def non_negative(
    scale: float = 1.0,
    dtype = np.float32,
):
    """Builds an initializer that returns an array
   with only positive values

    Args:
        scale: optional; scale of the normal distribution.
        dtype: optional; the initializer's default dtype.

    Returns:
        An initializer that returns arrays whose values are distributed like absolute
        values of a gaussian distribution.
    """

    def init(key,
            shape,
            dtype = dtype,
            out_sharding = None):
        
        dtype = jax.dtypes.canonicalize_dtype(dtype)

        return np.array(scale, dtype) * np.clip(jax.random.normal(key, shape).astype(dtype), min=0)
    
    return init

def simplex(
    scale: float = 1.0,
    dtype = np.float32,
):
    """
    Return an initializers that return random NON-NEGATIVE weights with a simplex constraint (i.e., L1 norm equal to 1)

    Args:
        scale: optional; scale of the simplex/L1-norm value (default to 1).
        dtype: optional; the initializer's default dtype.
    """


    def init(key,
            shape,
            dtype = dtype,
            out_sharding = None):
        
        dtype = jax.dtypes.canonicalize_dtype(dtype)

        # initialize random non-negative vectors
        weights = np.abs(jax.random.normal(key, shape).astype(dtype))

        # project to have L1 norm equal to scale ON THE FIRST AXIS
        # which in Flax-like convention is the one related to input dimension
        weights = weights * scale / np.sum(weights, axis=0, keepdims=True)

        return 
    
    return init

def time_conv_boolean_mask(
    dtype = np.bool,
):
    """
    Return an initializers that returns a boolean mask
    where we randomly select only 1 non-zero value for each element on the axis dimension
    """

    def init(key,
            shape,
            axis=0,
            dtype = dtype,
            out_sharding = None):
        
        dtype = jax.dtypes.canonicalize_dtype(dtype)

        idx_shape = shape[:axis] + shape[axis+1:]
        idx = jax.random.randint(key, idx_shape, 0, shape[axis])
        
        # return a mask of given shape,
        # setting True the values at (idx[i,j,...], i, j, ...)
        mask = jax.nn.one_hot(idx, shape[axis], axis=axis).astype(dtype)
        
        return mask
    
    return init