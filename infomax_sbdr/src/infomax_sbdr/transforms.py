def minmax_transform(
    x,
    in_min,
    in_max,
    out_min,
    out_max,
):
    """
    Transform x in range [in_min, in_max] to range [out_min, out_max]

    Args:
        x: input data
        in_min: lower bound of input range, broadcast-compatible with x
        in_max: upper bound of input range, broadcast-compatible with x
        out_min: lower bound of output range, broadcast-compatible with x
        out_max: upper bound of output range, broadcast-compatible with x

    Returns:
        transformed x
    """
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def offsetscale_transform(x, offset, scale):
    """
    Transform x by subtracting offset and dividing by scale

    Args:
        x: input data
        offset: offset to subtract from x, broadcast-compatible with x
        scale: scale to divide x by, broadcast-compatible with x

    Returns:
        transformed x
    """
    return (x - offset) / scale
