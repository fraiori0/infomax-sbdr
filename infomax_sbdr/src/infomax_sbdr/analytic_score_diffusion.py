import os
import jax
import jax.numpy as np
from jax import jit, vmap
from functools import partial


@jit
def cosine_schedule(t: float) -> float:
    """Cosine noise schedule."""
    s = 0.008
    f_t = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    return f_t / f_0

@jit
def linear_schedule(t: float) -> float:
    """Linear noise schedule."""
    return 1.0 - t


@jit
def compute_posterior_weights(
    phi: np.ndarray,        # Current state (*batch_dims, features)
    ks: np.ndarray,         # Data points (*batch_dims, k, features)
    alpha_bar_t: float      # Noise schedule value
) -> np.ndarray:
    """
    Compute posterior weights W_t(φ|ϕ) for all training points.

    Returns:
        Posterior weights, shape (batch_size, num_train)
    """
    # Compute shrunk means: √ᾱ_t φ
    means = np.sqrt(alpha_bar_t) * ks  # (*batch_dims, k, features)
    var = 1.0 - alpha_bar_t

    # Compute squared distances: ||ϕ - √ᾱ_t φ||²
    diff = phi[..., None, :] - means  # (*batch_dims, k, features)
    sq_dist = np.sum(diff ** 2, axis=-1)       # (*batch_dims, k)

    # Compute log-likelihoods (unnormalized)
    log_likelihoods = -0.5 * sq_dist / var # (*batch_dims, k)

    # Normalize using jax.nn log-sum-exp for numerical stability
    log_weights = log_likelihoods - jax.scipy.special.logsumexp(
        log_likelihoods, axis=-1, keepdims=True
    )

    return np.exp(log_weights)

@jit
def ideal_score(
    phi: np.ndarray,
    ks: np.ndarray,
    alpha_bar_t: float
) -> np.ndarray:
    """
    Compute the ideal score function analytically.

    Returns:
        Score estimates, shape (*batch_dims, features)
    """
    # Compute posterior weights
    weights = compute_posterior_weights(phi, ks, alpha_bar_t) # (*batch_dims, k)

    # Compute shrunk training points
    shrunk_data = np.sqrt(alpha_bar_t) * ks # (*batch_dims, k, features)

    # Compute differences: ϕ - √ᾱ_t φ
    diff = phi[..., None, :] - shrunk_data # (*batch_dims, k, features)

    # Weighted sum
    weighted_diff = weights[..., None] * diff # (*batch_dims, k, features)
    sum_weighted_diff = np.sum(weighted_diff, axis=-2) # (*batch_dims, features)

    # Score: (1/(1-ᾱ_t)) * Σ (√ᾱ_t φ - ϕ) W_t
    # Which equals: -(1/(1-ᾱ_t)) * Σ (ϕ - √ᾱ_t φ) W_t
    variance = 1.0 - alpha_bar_t 
    score = -sum_weighted_diff / variance # (*batch_dims, features)

    return score


@partial(jit, static_argnums=(1,))
def gamma_fn(t: float, alpha_bar_fn=cosine_schedule) -> float:
    """Compute γ_t = -∂_t ᾱ_t / (2 ᾱ_t) using finite differences."""
    dt = 1e-5
    alpha_t = alpha_bar_fn(t)
    alpha_t_plus = alpha_bar_fn(t + dt)
    d_alpha = (alpha_t_plus - alpha_t) / dt
    gamma = -d_alpha / (2.0 * alpha_t + 1e-8)
    return gamma

@partial(jit, static_argnums=(4,))
def reverse_diffusion_step(
    phi_t: np.ndarray, # (*batch_dims, features)
    ks: np.ndarray, # (*batch_dims, k, features)
    t: float,
    dt: float,
    alpha_bar_fn=cosine_schedule,
):
    """
    Single step of reverse diffusion.

    Returns:
        - Next state ϕ_{t-dt}
        - Statistics dictionary
    """
    alpha_bar_t = alpha_bar_fn(t)
    score_t = ideal_score(phi_t, ks, alpha_bar_t) # (*batch_dims, features)
    gamma_t = gamma_fn(t)

    # CORRECTED UPDATE: Note the PLUS sign!
    # ϕ_{t-dt} = ϕ_t + γ_t·dt·(ϕ_t + s_t(ϕ_t))
    phi_t_minus_dt = phi_t + gamma_t * dt * (phi_t + score_t) # (*batch_dims, features)

    # Collect statistics
    aux = {
        't': t,
        'alpha_bar': alpha_bar_t,
        'gamma': gamma_t,
        'mean': np.mean(phi_t),
        'std': np.std(phi_t),
        'min': np.min(phi_t),
        'max': np.max(phi_t),
        'score_norm': np.mean(np.linalg.norm(score_t, axis=-1)),
    }

    return phi_t_minus_dt, aux



def diffuse_with_subsets(
    key: jax.random.PRNGKey,
    phi_T: np.ndarray,
    ks: np.ndarray,
    T_max: float = 1.0,
    num_steps: int = 100,
):
    """
    Generate samples using subset-based ideal score machine, computing an analytic score function
    from the given points (ks).

    Args:
        key: Random key
        ks: Array of shape (*batch_dims, features), points to use to compute the analytic score function
        num_steps: Number of diffusion steps
        T_max: Maximum diffusion time

    Returns:
        - Trajectory: array of shape (*batch_dims, num_steps+1, features)
        - Statistics: list of stat dictionaries for each step
    """

    # Time discretization
    dt = T_max / num_steps
    times = np.linspace(T_max, 0, num_steps + 1)

    # Define a function to scan over time with jax.lax.scan
    def f_scan(carry, input):
        phi_t = carry # (*batch_dims, features)
        t = input
        phi_t, stats = reverse_diffusion_step(phi_t, ks, t, dt)
        return phi_t, (phi_t, stats)

    # Scan over time
    phi_0, (trajectory, statistics) = jax.lax.scan(
        f_scan,
        phi_T,
        times,
    )

    return trajectory, statistics
