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


@jit
def expected_Jaccard_index_stable(px1, px2, eps=1e-8):
    """Expected Jaccard index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli
        eps (float, optional): small epsilon added to each probability to avoid division by zero. Defaults to 1e-8.

    Returns:
        (ndarray): expected Jaccard index
    """
    return (((px1 + eps) * (px2 + eps))).sum(axis=-1) / (
        1 - (1 - (px1 + eps)) * (1 - (px2 + eps))
    ).sum(axis=-1)


@jit
def expected_Jaccard_index(px1, px2, eps=1e-8):
    """Expected Jaccard index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli
        eps (float, optional): small epsilon added to each probability to avoid division by zero. Defaults to 1e-8.

    Returns:
        (ndarray): expected Jaccard index
    """
    return ((px1 * px2)).sum(axis=-1) / (1 - (1 - px1) * (1 - px2) + eps).sum(axis=-1)


@jit
def expected_corrected_Jaccard_index(px1, px2):
    # WORKS <--- USE THIS
    """Expected Jaccard index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli
        eps (float, optional): small epsilon added to each probability to avoid division by zero. Defaults to 1e-8.

    Returns:
        (ndarray): expected Jaccard index
    """
    return (((px1 * px2)).sum(axis=-1) + 1.0) / (
        (1 - (1 - px1) * (1 - px2)).sum(axis=-1) + 1.0
    )


@jit
def expected_corrected_Jaccard_index2(px1, px2):
    """Expected Jaccard index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli
        eps (float, optional): small epsilon added to each probability to avoid division by zero. Defaults to 1e-8.

    Returns:
        (ndarray): expected Jaccard index
    """
    return (((px1 * px2)).sum(axis=-1) + 0.1) / (
        (1 - (1 - px1) * (1 - px2)).sum(axis=-1) + 0.1
    )


@jit
def expected_overlap_index(px1, px2):
    """Expected overlap index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected overlap index
    """
    return (px1 * px2).sum(axis=-1) / np.minimum(px1.sum(axis=-1), px2.sum(axis=-1))


@jit
def expected_overlap_index_stable(px1, px2, eps=1e-8):
    """Expected overlap index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected overlap index
    """
    return ((px1 + eps) * (px2 + eps)).sum(axis=-1) / np.minimum(
        (px1 + eps).sum(axis=-1), (px2 + eps).sum(axis=-1)
    )


@jit
def expected_corrected_overlap_index(px1, px2):
    # WORKS <--- USE THIS
    """Expected overlap index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected overlap index
    """
    return ((px1 * px2).sum(axis=-1) + 1.0) / (
        np.minimum(px1.sum(axis=-1), px2.sum(axis=-1)) + 1.0
    )


@jit
def expected_corrected_overlap_index2(px1, px2):
    """Expected overlap index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected overlap index
    """
    return ((px1 * px2).sum(axis=-1) + 0.1) / (
        np.minimum(px1.sum(axis=-1), px2.sum(axis=-1)) + 0.1
    )


@jit
def expected_overlap_index_feature(px1, px2):
    """Expected overlap index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected overlap index
    """
    return ((px1 * px2) / np.minimum(px1, px2)).sum(axis=-1)


@jit
def expected_coincidence_index(px1, px2):
    """Expected coincidence index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli

    Returns:
        (ndarray): expected coincidence index
    """
    return expected_Jaccard_index(px1, px2) * expected_overlap_index(px1, px2)


@jit
def expected_averaged_coincidence_index(px1, px2, kj=0.5):
    """Expected averaged coincidence index between two samples taken from the input from the input multivariate Bernoulli distirbutions

    Args:
        px1 (ndarray): probability of activation of the first multivariate Bernoulli
        px2 (ndarray): probability of activation of the second multivariate Bernoulli
        kj (float, optional): weight of the Jaccard index in the averaged coincidence index. Defaults to 0.5.

    Returns:
        (ndarray): expected averaged coincidence index
    """
    return kj * expected_Jaccard_index(px1, px2) + (1.0 - kj) * expected_overlap_index(
        px1, px2
    )


@jit
def expected_sorensen_index(px1, px2, eps=1e-6):
    """Expected Sorensen index between two samples taken from the input multivariate Bernoulli distributions"""
    return (2.0 * (px1 * px2).sum(axis=-1) + eps) / (
        (px1 * px1).sum(axis=-1) + (px2 * px2).sum(axis=-1) + eps
    )


@jit
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


@jit
def bernoulli_entropy_stable(p):
    """Compute the entropy of a multivariate Bernoulli distribution
    Args:
        p (ndarray): probabilities of being =1 of each element of the multivariate bernoulli distribution

    Returns:
        (ndarray): entropy
    """
    return -(p * np.log(p + 1e-6) + (1 - p) * np.log(1 - p + 1e-6)).sum(axis=-1)


@jit
def bernoulli_crossentropy_stable(p, q):
    """Compute the cross-entropy between two multivariate Bernoulli distribution
    Args:
        p (ndarray): probabilities of being =1 of each element of the first multivariate bernoulli distribution
        q (ndarray): probabilities of being =1 of each element of the second multivariate bernoulli distribution

    Returns:
        (ndarray): cross-entropy
    """
    return -(p * np.log(q + 1e-6) + (1 - p) * np.log(1 - q + 1e-6)).sum(axis=-1)


@jit
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


@jit
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


@jit
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


@jit
def proxy_jaccard_index(px1, px2, eps=1.0):
    """
    p(px1 | px2) = E(AND(px1, px2)) / E(OR(px1, px2))
    """
    return (expected_and(px1, px2) + eps) / (expected_or(px1, px2) + eps)


@jit
def asymmetric_jaccard_similarity(px1, px2, eps=1.0e-6):
    """
    p(px1 | px2) = E(AND(px1, px2)) / E(OR(px2, px2))
    """
    return (expected_and(px1, px2) + eps) / (expected_or(px2, px2) + eps)


@jit
def expected_custom_index(px1, px2, eps=1.0e-6):
    return expected_and(px1, px2) / (px2.sum(axis=-1) + eps)


@jit
def gamma_similarity(px1, px2, eps=1.0e-6):
    """
    p(px1 | px2) = E(AND(px1, px2)) / E(|px2|)
    """
    return expected_and(px1, px2) / (px2.sum(axis=-1) + eps)


@jit
def active_crossentropy(px1, px2, eps=1.0e-6):
    return -(px1 * np.log(px2 + eps)).sum(axis=-1)


@jit
def active_kl_divergence(px1, px2, eps=1.0e-6):
    return (px1 * (np.log(px1 + eps) - np.log(px2 + eps))).sum(axis=-1)


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


@jit
def sr_similarity_index(sr1, sr2):
    """Use on already normalized sr1 and sr2"""
    return 2 * (sr1 * sr2).sum(axis=-1) / (sr1 + sr2).sum(axis=-1)


@jit
def poisson_kl_divergence_stable(p, q, eps=1e-6):
    """Compute the Kullback-Leiber Divergence between two Poisson distributions DKL(p||q)
    Args:
        p (ndarray): parameters of the first Poisson distribution
        q (ndarray): parameters of the second Poisson distribution

    Returns:
        (ndarray): Kullback-Leiber Divergence
    """
    return (p * np.log((p + eps) / (q + eps) + eps) - p + q).sum(axis=-1)


@jit
def estimate_mi(ys):
    """Estimate the Mutual Information using the given p_ij matrix"""

    # estimate InfoNCE bound on MI (as from FLO paper)

    # compute the value of the critic
    # g(x,y) = ln(p_ij[i,j]) with p_ij[i,j] = p(y_i | y_j)
    p_ij = expected_custom_index(ys[..., :, None, :], ys[..., None, :, :])
    # p_ij = sdm.expected_Jaccard_index(ys[:, None, :], ys[None, :, :])
    # p_ij = sdm.expected_and(ys[:, None, :], ys[None, :, :])

    # # evaluate the partition function associated with the tilting function
    # the exponent is removed because the critic g(x,y) has logarithm, they cancel out
    # TODO: should this be the mean or the sum? check FLO paper
    # should be correct with the mean, re-check before publication (see appendix page 3 where they reach Eq. 16)
    z_i = p_ij.mean(axis=-1)

    pmi = np.log((np.diag(p_ij) / (z_i + 1e-6)) + 1e-6)
    # pmi = np.log(1.0/(z_i + 1e-6))
    # but if ys has zero active units, the pmi is actually 0
    pmi = pmi * (ys.sum(axis=-1) > 1e-6)

    MI = pmi.mean(axis=-1)

    return MI
