import jax
import jax.numpy as np
from jax import jit


def flo(uii, pii, pij, eps=1e-6):
    """Estimate the Mutual Information between x_i and y_i using the FLO estimator

    Args:
        uii (np.ndarray): -log( p(x_i, y_i) / ( p(x_i)p(y_i) ) )
        pii (np.ndarray): p(x_i | y_i). Note, no logarithm here, as we would anyway raise it using exponentials and cancel it out
        pij (np.ndarray): p(x_i | y_j). Note, no logarithm here, as we would anyway raise it using exponentials and cancel it out
        eps (float, optional): Small value to add in operations that could be numerically unstable. Defaults to 1e-6.

    Returns:
        (np.ndarray): Mutual Information between x_i and y_i

    Note:
        Here we consider that pij does contain pii
    """
    # e_p = ((pij) / (pii + eps)).mean(axis=0)

    # if pii is also contained in pij, we need to subtract 1
    e_p = (1.0 / (pii.shape[-1] - 1.0)) * (
        ((pij) / (pii + eps)[..., None]).sum(axis=-1) - 1.0
    )

    flo = 1 - (uii + np.exp(-uii) * e_p)

    return flo


def flo_original(uii, gii, gij, eps=1e-6):
    """Estimate the Mutual Information between x_i and y_i using the FLO estimator

    Args:
        uii (np.ndarray): neg_pmi (x_i, y_y)
        gii (np.ndarray): critic g(x_i, y_i)
        gij (np.ndarray): critic g(x_i, y_j)
        eps (float, optional): Small value to add in operations that could be numerically unstable. Defaults to 1e-6.

    Returns:
        (np.ndarray): Mutual Information between x_i and y_i

    Note:
        Here we consider that pij does contain pii (in the diagonal)
    """
    # e_p = ((pij) / (pii + eps)).mean(axis=0)

    # if pii is also contained in pij, we need to subtract 1
    e_p = (1.0 / (gii.shape[-1] - 1.0)) * (
        np.exp(gij - gii[..., None]).sum(axis=-1) - 1.0
    )

    flo = 1 - (uii + np.exp(-uii) * e_p)

    return flo


def weighted_flo(uii, pii, pij, wij, eps=1e-6):
    """Estimate the Weighted Mutual Information between x_i and y_i using the FLO estimator

    Args:
        uii (np.ndarray): -log( p(x_i, y_i) / ( p(x_i)p(y_i) ) )
        pii (np.ndarray): p(x_i | y_i). Note, no logarithm here, as we would anyway raise it using exponentials and cancel it out
        pij (np.ndarray): p(x_i | y_j). Note, no logarithm here, as we would anyway raise it using exponentials and cancel it out
        eps (float, optional): Small value to add in operations that could be numerically unstable. Defaults to 1e-6.

    Returns:
        (np.ndarray): Weighted Mutual Information between x_i and y_i

    Note:
        Here we consider that pij does NOT contain pii, and wij does NOT contain wii
    """
    # ws_self is assumed to be equal to 1
    e_p = (1.0 / (wij.sum(axis=0) - 1.0)) * (
        (wij * (pij) / (pii + eps)).sum(axis=0) - 1.0
    )

    # # FLO
    flo = 1 - (uii + np.exp(-uii) * e_p)

    return flo
