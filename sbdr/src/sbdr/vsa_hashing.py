import jax
import jax.numpy as np
from jax import jit, grad, vmap
from functools import partial


def generate_random_vectors(key, n_projections, n_dimensions):
    """Generate a batch of random projection vectors each with size (n_features).
    Output is of size (n_projections, n_features)
    """
    Vs = jax.random.normal(key, shape=(n_projections, n_dimensions))
    # normalize
    return Vs / np.linalg.norm(Vs, axis=-1, keepdims=True)


def projection_to_hash_gen(p, n_bins: int):
    """Generate a hash from a projection p."""
    # take p mod n_bins
    # (note: if p is in the range [0, n_bins] then p_idx is the integer part of p)
    p_idx = np.mod(p, n_bins).astype(int)
    # create a one-hot vector of size n_bins
    h = np.zeros(n_bins).astype(int)
    # set the p_idx-th element of h to 1
    h = h.at[p_idx].set(1)
    return h


def hash_gen(x, Vs, range_min, range_max, n_bins: int):
    """Generate a hash from a vector x using the random projection matrix V.
    x is of size (n_dimensions)
    V is of size (n_projections, n_dimensions)
    Output is of size (n_projections, n_bins)
    """
    # rescale x so that the norm of each vector is in the range [0, n_bins)
    x = x / np.linalg.norm(range_max - range_min) * (n_bins - 1e-1)
    # x = (x - range_min) / (range_max - range_min) * n_bins
    # project the vector x with the matrix V
    ps = np.dot(Vs, x)
    # apply the hash function to the projected vector
    project = vmap(
        partial(
            projection_to_hash_gen,
            n_bins=n_bins,
        ),
        in_axes=0,
        out_axes=0,
    )
    hs = project(ps)
    return hs.reshape(-1)


def get_batch_hash_function(
    key, n_projections, n_dimensions, range_min, range_max, n_bins
):
    """Generate a function that can be used to hash a batch of vectors."""
    # Generate the random (unit) vectors used to project the data
    key, _ = jax.random.split(key, 2)
    Vs = generate_random_vectors(key, n_projections, n_dimensions)
    # Generate the hash function
    batch_hash = jit(
        vmap(
            partial(
                hash_gen,
                Vs=Vs,
                range_min=range_min,
                range_max=range_max,
                n_bins=n_bins,
            ),
            in_axes=0,
            out_axes=0,
        )
    )
    return batch_hash


if __name__ == "__main__":
    key = jax.random.PRNGKey(5)

    n_dimensions = 3
    n_projections = 20
    n_bins = 8
    n_batch = 200

    range_min = np.array((-15.0, -15.0, -15.0))
    range_max = np.array((15.0, 15.0, 15.0))

    # Generate random points, to be hashed
    key, _ = jax.random.split(key, 2)
    xs = jax.random.uniform(
        key,
        shape=(n_batch, n_dimensions),
        minval=range_min[None,],
        maxval=range_max[None,],
    )

    # Generate the hashing function
    # (this will also generate the random vectors used to project the data)
    batch_hash = get_batch_hash_function(
        key, n_projections, n_dimensions, range_min, range_max, n_bins
    )

    ys = batch_hash(xs)

    # COmpute euclidean distance between original points
    # NOTE: consider the original point as already normalized in [0, 1] over each axis
    xs_norm = (xs - range_min[None]) / (range_max[None] - range_min[None])
    ds_xs = np.linalg.norm(xs_norm[:, None, :] - xs_norm[None, :, :], axis=-1).reshape(
        -1
    )

    # And between hashed points
    from binary_comparisons import tmp_index

    ds_ys = -tmp_index(ys[:, None, :], ys[None, :, :], eps=1e-2).reshape(-1)
    # ds_ys = np.abs(ys[:, None, :] - ys[None, :, :]).sum(axis=-1).reshape(-1)

    # plot
    import plotly.graph_objects as go

    # plot data in a Histogram
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ds_xs,
            y=ds_ys,
            mode="markers",
            name="Similarity vs Distance",
            marker_color="#c41230",
            marker=dict(
                line=dict(width=1.0, color="#000000"),
            ),
        )
    )
    fig.update_layout(
        title="Distribution of similarities between outputs",
        xaxis_title="Euclidean Distance",
        yaxis_title="Jaccard Distance",
    )
    fig.show()
