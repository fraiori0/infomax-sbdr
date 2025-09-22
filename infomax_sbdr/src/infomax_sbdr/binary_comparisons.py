import jax
import jax.numpy as np
from jax import jit


def and_soft(px1, px2):
    return px1 * px2


def or_soft(px1, px2):
    return 1 - (1 - px1) * (1 - px2)


def xor_soft(px1, px2):
    return (1 - px1) * px2 + px1 * (1 - px2)


def andnot_soft(px1, px2):
    return px1 * (1 - px2)


def bernoulli_kl_divergence_stable(p, q):
    """Compute the Kullback-Leiber Divergence between two multivariate  DKL(p||q)
    Args:
        p (ndarray): probabilities of being =1 of each element of the first multivariate bernoulli distribution
        q (ndarray): probabilities of being =1 of each element of the second multivariate bernoulli distribution

    Returns:
        (ndarray): Kullback-Leiber Divergence
    """
    return (
        p * np.log((p + 1e-6) / (q + 1e-6))
        + (1 - p) * np.log((1 - p + 1e-6) / (1 - q + 1e-6))
    ).sum(axis=-1)


def bernoulli_entropy_stable(p):
    """Compute the entropy of a multivariate Bernoulli distribution
    Args:
        p (ndarray): probabilities of being =1 of each element of the multivariate bernoulli distribution

    Returns:
        (ndarray): entropy
    """
    return -(p * np.log(p + 1e-6) + (1 - p) * np.log(1 - p + 1e-6)).sum(axis=-1)


def negative_bernoulli_crossentropy_stable(p, q, eps=1e-6):
    """Compute the cross-entropy between two multivariate Bernoulli distribution
    Args:
        p (ndarray): probabilities of being =1 of each element of the first multivariate bernoulli distribution
        q (ndarray): probabilities of being =1 of each element of the second multivariate bernoulli distribution

    Returns:
        (ndarray): cross-entropy

    Warning:
        This is the negative of the crossentropy, not the crossentropy. The higher the negative crossentropy, the more similar the two distributions.
    """
    return (p * np.log(q + eps) + (1 - p) * np.log(1 - q + eps)).sum(axis=-1)


def expected_and(px1, px2):
    """Expected amount of active bits if we take a sample from each of the two multivariate Bernoulli and perform a AND operation
        between the two samples

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected amount of active bits
    """
    return (px1 * px2).sum(axis=-1)


def expected_or(px1, px2):
    """Expected amount of active bits if we take a sample from each of the two multivariate Bernoulli and perform a OR operation
        between the two samples

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected amount of active bits
    """
    return (1 - (1 - px1) * (1 - px2)).sum(axis=-1)


def expected_xor(px1, px2):
    """Expected amount of active bits if we take a sample from each of the two multivariate Bernoulli and perform a XOR operation
        between the two samples

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected amount of active bits
    """
    return ((1 - px1) * px2 + px1 * (1 - px2)).sum(axis=-1)


def jaccard_index(px1, px2, eps=1.0):
    """
    p(px1 | px2) = E(AND(px1, px2)) / E(OR(px1, px2))
    """
    return (expected_and(px1, px2) + eps) / (expected_or(px1, px2) + eps)


def jaccard_index_mod(px1, px2, eps=1.0):
    """
    p(px1 | px2) = E(AND(px1, px2)) / E(OR(px1, px2))
    """
    return (expected_and(px1, px2)) / (expected_or(px1, px2) + eps)


def asymmetric_jaccard_index(px1, px2, eps=1.0e-2):
    """
    p(px1 | px2) = E(AND(px1, px2)) / E(|px2|)
    """
    return (expected_and(px1, px2) + eps) / (px2.sum(axis=-1) + eps)

def log_and(px1, px2, eps=1.0):
    # note, here we assume an implicit log operation is performed on the output of this is function,
    # which cancels out with the exponentiation in the FLO estimator
    return expected_and(px1, px2) + eps


def active_crossentropy(px1, px2, eps=1.0e-6):
    return -(px1 * np.log(px2 + eps)).sum(axis=-1)


def active_kl_divergence(px1, px2, eps=1.0e-6):
    return (px1 * (np.log(px1 + eps) - np.log(px2 + eps))).sum(axis=-1)


def cosine_similarity_normalized(x1, x2, eps=1.0e-6):
    x1_normalized = x1 / (np.linalg.norm(x1, axis=-1, keepdims=True) + eps)
    x2_normalized = x2 / (np.linalg.norm(x2, axis=-1, keepdims=True) + eps)
    return (x1_normalized * x2_normalized).sum(axis=-1)


def circulant(v: np.ndarray) -> np.ndarray:
    """
    Create a circulant matrix from a 1D array.
    :param v: 1D array
    :return: Circulant matrix
    """
    n = len(v)
    c = np.zeros((n, n))
    c = c.at[0].set(v)
    for i in range(1, n):
        c = c.at[i].set(np.roll(v, i))
    return c


def poisson_kl_divergence_stable(p, q, eps=1e-6):
    """Compute the Kullback-Leiber Divergence between two Poisson distributions DKL(p||q)
    Args:
        p (ndarray): parameters of the first Poisson distribution
        q (ndarray): parameters of the second Poisson distribution

    Returns:
        (ndarray): Kullback-Leiber Divergence
    """
    return (p * np.log((p + eps) / (q + eps) + eps) - p + q).sum(axis=-1)
