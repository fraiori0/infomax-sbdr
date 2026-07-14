"""
Contrastive SBDR Dictionary Learning — synthetic benchmark
============================================================

Compares two mechanisms for inducing sparse, almost-binary, distributed
codes in a dictionary-learning setting:

  1. "ours": codes z in [0,1]^dz trained with

        L(D, Z) = ||x - D^T z||^2
                  - lam * log(<z, z> + eps)
                  + lam * log(<z, z_bar> + eps),      z_bar = mean_j z_j

     i.e. reconstruction plus the InfoNCE-style contrastive term built from
     the logarithmic dot-product critic g_eps(u,v) = log(<u,v> + eps), using
     the *exact* algebraic simplification exp(g_eps(z_i,z_j)) = <z_i,z_j>+eps,
     which lets the average over negatives be replaced by a dot product with
     the population mean code z_bar (no Jensen gap, unlike a softmax/dot
     critic). This replaces L1 regularization as the sparsity mechanism.

  2. "l1_box" / "l1_free": classical ISTA-based sparse dictionary learning
     with L1 regularization, in a box-constrained ([0,1]^dz, fair comparison
     to "ours") and a fully unconstrained (classical) variant.

Both use the same alternating-minimization skeleton:

    for outer_step in range(T):
        Z <- several inner Adam + projection/soft-threshold steps, D fixed
             (for "ours": z_bar is recomputed at *every* inner step from the
              live Z -- this "self-consistent field" recomputation is
              necessary; freezing z_bar changes the optimum.)
        D <- closed-form ridge least-squares solve, Z fixed

NOTE on the optimizer: code updates use Adam rather than a fixed
Lipschitz-bound step size. Reason: the contrastive term's restoring force
-2*lam*z/u (u = <z,z>+eps) *weakens* as codes get denser (larger u), which
can trap free (non-amortized) per-sample codes in a dense local optimum once
reconstruction pressure has pushed them there -- exactly the state where a
weak-but-consistently-signed gradient needs a normalized (not
worst-case-bounded) step size to still make progress. `lipschitz_recon` /
`lipschitz_ours` are kept below as analytic reference/diagnostics, not used
for step sizing.

Held-out inference reuses the identical per-sample solver with D and (for
"ours") a *frozen* population statistic z_bar_ref taken from the training
set -- analogous to using running statistics instead of live batch
statistics at eval time in BatchNorm.

Requirements: jax, jaxlib, numpy, scipy, matplotlib.
Usage:
    python sbdr_dictionary_learning.py            # full run
    python sbdr_dictionary_learning.py --quick    # tiny smoke test
"""

from __future__ import annotations

import os
import time
import argparse
import collections
from dataclasses import dataclass, replace

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Improves numerical stability of the log-ratio contrastive term.
jax.config.update("jax_enable_x64", True)


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class DataConfig:
    d_x: int = 48
    d_z: int = 96
    n_groups: int = 200          # number of distinct Beta-sampled activation profiles
    m_per_group: int = 10        # samples drawn per profile -> n_train = n_groups*m_per_group
    n_test_groups: int = 40
    sparsity_mean: float = 0.10  # mu: expected fraction of active units per sample
    concentration: float = 50.0  # c: beta concentration (higher = more uniform unit usage)
    a_min: float = 0.85          # sharpness: active-unit values ~ U(a_min, 1)
    coherence: float = 0.0       # 0 = incoherent dictionary atoms, -> 1 highly coherent
    snr_db: float = 20.0
    seed: int = 0


@dataclass
class OursConfig:
    epsilon: float = 1e-2
    lam: float = 0.15
    code_inner_steps: int = 25
    outer_steps: int = 30
    code_lr: float = 0.05        # Adam learning rate for the code updates
    ridge: float = 1e-4          # D-step ridge regularization


@dataclass
class ISTAConfig:
    alpha: float = 0.02          # soft-threshold magnitude applied after each Adam step
    code_inner_steps: int = 25
    outer_steps: int = 30
    code_lr: float = 0.05
    ridge: float = 1e-4
    box_constrained: bool = True  # False -> classical unconstrained ISTA


# ============================================================
# 2. Synthetic data generation
# ============================================================

def sample_sbdr_codes(key, n_groups, m_per_group, d_z, mu, concentration, a_min):
    """Ground-truth SBDR codes via the Beta/Bernoulli/Uniform procedure:
    (1) draw n_groups activation-probability profiles theta ~ Beta(mu*c, (1-mu)*c),
    (2) for each profile draw m_per_group binary masks ~ Bernoulli(theta),
    (3) set active-unit values ~ Uniform(a_min, 1).
    Returns an array of shape (n_groups*m_per_group, d_z) in [0,1]^d_z.
    """
    key_theta, key_mask, key_val = random.split(key, 3)
    a_beta = mu * concentration
    b_beta = (1.0 - mu) * concentration
    theta = random.beta(key_theta, a_beta, b_beta, shape=(n_groups, d_z))
    theta_rep = jnp.repeat(theta, m_per_group, axis=0)
    n = n_groups * m_per_group
    mask = random.bernoulli(key_mask, theta_rep)
    values = random.uniform(key_val, shape=(n, d_z), minval=a_min, maxval=1.0)
    return mask * values


def sample_dictionary(key, d_z, d_x, coherence):
    """Random unit-norm dictionary atoms (rows) with a controllable coherence
    knob: coherence=0 -> independent random directions; coherence->1 -> all
    atoms collapse onto a shared random direction."""
    key_atoms, key_common = random.split(key)
    atoms = random.normal(key_atoms, shape=(d_z, d_x))
    common = random.normal(key_common, shape=(d_x,))
    common = common / jnp.linalg.norm(common)
    atom_norms = jnp.linalg.norm(atoms, axis=1, keepdims=True)
    mixed = (1.0 - coherence) * atoms + coherence * common[None, :] * atom_norms
    return mixed / jnp.linalg.norm(mixed, axis=1, keepdims=True)


def generate_dataset(cfg: DataConfig):
    key = random.PRNGKey(cfg.seed)
    k_train_z, k_test_z, k_dict, k_noise_tr, k_noise_te = random.split(key, 5)

    Z_train = sample_sbdr_codes(k_train_z, cfg.n_groups, cfg.m_per_group,
                                 cfg.d_z, cfg.sparsity_mean, cfg.concentration, cfg.a_min)
    Z_test = sample_sbdr_codes(k_test_z, cfg.n_test_groups, cfg.m_per_group,
                                cfg.d_z, cfg.sparsity_mean, cfg.concentration, cfg.a_min)
    D_true = sample_dictionary(k_dict, cfg.d_z, cfg.d_x, cfg.coherence)

    def add_noise(Z, key):
        clean = Z @ D_true
        signal_power = jnp.mean(clean ** 2)
        noise_std = jnp.sqrt(signal_power / (10 ** (cfg.snr_db / 10.0)))
        return clean + noise_std * random.normal(key, clean.shape)

    X_train = add_noise(Z_train, k_noise_tr)
    X_test = add_noise(Z_test, k_noise_te)
    return dict(D_true=D_true, Z_train=Z_train, X_train=X_train, Z_test=Z_test, X_test=X_test)


# ============================================================
# 3. Core objectives (D: (d_z, d_x), z: (d_z,), x: (d_x,))
# ============================================================

def reconstruction_loss(z, x, D):
    diff = x - z @ D
    return jnp.dot(diff, diff)


def contrastive_term(z, z_bar, epsilon):
    """-log((<z,z>+eps) / (<z,z_bar>+eps)) -- the exact averaged-negative
    simplification of the InfoNCE bound under the g_eps critic."""
    u = jnp.dot(z, z) + epsilon
    v = jnp.dot(z, z_bar) + epsilon
    return -jnp.log(u) + jnp.log(v)


def ours_code_objective(z, x, D, z_bar, epsilon, lam):
    return reconstruction_loss(z, x, D) + lam * contrastive_term(z, z_bar, epsilon)


def recon_only_objective(z, x, D):
    """Smooth part of the ISTA objective; the L1 term is handled separately
    via the proximal (soft-threshold) operator."""
    return reconstruction_loss(z, x, D)


def ours_batch_objective(Z, X, D, epsilon, lam):
    """Batch-level objective with z_bar computed *inside* the differentiated
    function. This is the version that must be used during training: since
    z_bar = mean(Z, axis=0) depends on every sample, autodiff through this
    function produces the *exact* mean-field-corrected gradient (verified
    against the closed-form correction derived earlier: grad_joint -
    grad_frozen is a single shared constant vector c/N added to every row,
    confirmed numerically to machine precision).

    Using vmap(grad(per_sample_objective)) with z_bar passed in as a
    separate, non-differentiated argument -- which is what an earlier
    version of this script did -- silently drops that correction: recomputing
    z_bar's *value* between optimizer steps is not the same as differentiating
    *through* its dependence on the current batch within a step. That bug
    produced a lambda-independent plateau once the contrastive term
    dominated (the surviving, wrong stationarity condition no longer
    contains lambda), matching what was observed empirically.
    """
    diff = X - Z @ D
    recon = jnp.sum(diff ** 2, axis=1)
    z_bar = jnp.mean(Z, axis=0)
    u = jnp.sum(Z * Z, axis=1) + epsilon
    v = Z @ z_bar + epsilon
    contrastive = -jnp.log(u) + jnp.log(v)
    return jnp.mean(recon + lam * contrastive)


_ours_joint_grad = jax.grad(ours_batch_objective, argnums=0)

# Per-sample gradient with z_bar as a genuine constant -- correct (not an
# approximation) for *inference*, where z_bar_ref is a frozen reference
# statistic that does not depend on the codes being optimized.
_ours_grad_single = jax.grad(ours_code_objective, argnums=0)
_ours_grad_batch = jax.vmap(_ours_grad_single, in_axes=(0, 0, None, None, None, None))

_recon_grad_single = jax.grad(recon_only_objective, argnums=0)
_recon_grad_batch = jax.vmap(_recon_grad_single, in_axes=(0, 0, None))


def lipschitz_recon(D):
    """Upper bound on the Lipschitz constant of the reconstruction gradient
    w.r.t. z: 2 * lambda_max(D D^T). Analytic reference/diagnostic only --
    NOT used for step sizing (see module docstring)."""
    gram = D @ D.T
    return 2.0 * jnp.linalg.eigvalsh(gram)[-1]


def lipschitz_ours(D, z_bar, epsilon, lam):
    """Worst-case (u=eps) curvature bound for the contrastive term, added to
    the reconstruction bound. Analytic reference/diagnostic only."""
    l_recon = lipschitz_recon(D)
    l_contrastive = 2.0 * lam / epsilon + lam * jnp.dot(z_bar, z_bar) / (epsilon ** 2)
    return l_recon + l_contrastive


def soft_threshold(v, t):
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t, 0.0)


# ============================================================
# 4. Generic Adam-optimized inner loop
# ============================================================

def adam_scan(grad_fn, Z0, n_inner, lr, project_fn, b1=0.9, b2=0.999, adam_eps=1e-8):
    """Runs n_inner Adam steps on Z (shape (n, d_z)), applying project_fn
    (box clip and/or soft-threshold) after every step.

    grad_fn(Z) -> array with Z's shape. Called with the *current* Z at every
    iteration, so any statistic grad_fn recomputes internally from Z (e.g.
    z_bar = mean(Z, axis=0)) is automatically "live" (self-consistent-field
    style), while a statistic captured in a closure before the loop starts
    stays frozen -- this is how the live-vs-frozen z_bar distinction between
    training and inference is implemented.
    """
    m0 = jnp.zeros_like(Z0)
    v0 = jnp.zeros_like(Z0)
    t0 = jnp.zeros((), dtype=Z0.dtype)

    def body(carry, _):
        Z, m, v, t = carry
        grads = grad_fn(Z)
        t = t + 1
        m = b1 * m + (1 - b1) * grads
        v = b2 * v + (1 - b2) * grads ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        Z_new = project_fn(Z - lr * m_hat / (jnp.sqrt(v_hat) + adam_eps))
        return (Z_new, m, v, t), None

    (Z_final, _, _, _), _ = jax.lax.scan(body, (Z0, m0, v0, t0), xs=None, length=n_inner)
    return Z_final


def ours_code_step(Z, X, D, epsilon, lam, n_inner, lr):
    """Training-time code update: uses the true joint gradient (z_bar
    differentiated through), which is required for correctness -- see
    ours_batch_objective's docstring."""
    grad_fn = lambda Zc: _ours_joint_grad(Zc, X, D, epsilon, lam)
    project_fn = lambda Zc: jnp.clip(Zc, 0.0, 1.0)
    return adam_scan(grad_fn, Z, n_inner, lr, project_fn)


def ista_code_step(Z, X, D, alpha, n_inner, lr, box_constrained):
    grad_fn = lambda Zc: _recon_grad_batch(Zc, X, D)
    if box_constrained:
        project_fn = lambda Zc: jnp.clip(soft_threshold(Zc, alpha), 0.0, 1.0)
    else:
        project_fn = lambda Zc: soft_threshold(Zc, alpha)
    return adam_scan(grad_fn, Z, n_inner, lr, project_fn)


def dictionary_step(Z, X, ridge):
    """Closed-form ridge-regularized least squares: D = (Z^T Z + ridge I)^-1 Z^T X."""
    d_z = Z.shape[1]
    A = Z.T @ Z + ridge * jnp.eye(d_z, dtype=Z.dtype)
    B = Z.T @ X
    return jnp.linalg.solve(A, B)


# ============================================================
# 5. Full training loops and held-out inference
# ============================================================

def _diagnostics(Z, epsilon):
    return dict(
        active=sparsity_stats(Z)["mean_active_units"],
        binarity=binarity_score(Z),
        mean_u=float(jnp.mean(jnp.sum(Z ** 2, axis=1) + epsilon)),
    )


def train_ours(X, cfg: OursConfig, d_z, key):
    n, d_x = X.shape
    k_D, k_Z = random.split(key)
    D = 0.1 * random.normal(k_D, (d_z, d_x)) / jnp.sqrt(d_z)
    Z = 0.05 * random.uniform(k_Z, (n, d_z))

    history = {"mse": [], "active": [], "binarity": [], "mean_u": []}
    for _ in range(cfg.outer_steps):
        Z = ours_code_step(Z, X, D, cfg.epsilon, cfg.lam, cfg.code_inner_steps, cfg.code_lr)
        D = dictionary_step(Z, X, cfg.ridge)
        history["mse"].append(float(jnp.mean((X - Z @ D) ** 2)))
        diag = _diagnostics(Z, cfg.epsilon)
        history["active"].append(diag["active"])
        history["binarity"].append(diag["binarity"])
        history["mean_u"].append(diag["mean_u"])
    return D, Z, history


def train_ista(X, cfg: ISTAConfig, d_z, key):
    n, d_x = X.shape
    k_D, k_Z = random.split(key)
    D = 0.1 * random.normal(k_D, (d_z, d_x)) / jnp.sqrt(d_z)
    Z = (0.05 * random.uniform(k_Z, (n, d_z)) if cfg.box_constrained
         else 0.05 * random.normal(k_Z, (n, d_z)))

    history = {"mse": [], "active": [], "binarity": []}
    for _ in range(cfg.outer_steps):
        Z = ista_code_step(Z, X, D, cfg.alpha, cfg.code_inner_steps, cfg.code_lr, cfg.box_constrained)
        D = dictionary_step(Z, X, cfg.ridge)
        history["mse"].append(float(jnp.mean((X - Z @ D) ** 2)))
        diag = _diagnostics(Z, 1e-2)
        history["active"].append(diag["active"])
        history["binarity"].append(diag["binarity"])
    return D, Z, history


def infer_ours(X_new, D, z_bar_ref, cfg: OursConfig, n_steps):
    """Encode new samples given a *frozen* D and a frozen reference z_bar
    (e.g. the training-set mean code) -- no batch/negative sampling needed."""
    n, d_z = X_new.shape[0], D.shape[0]
    Z0 = 0.05 * jnp.ones((n, d_z), dtype=X_new.dtype)
    grad_fn = lambda Zc: _ours_grad_batch(Zc, X_new, D, z_bar_ref, cfg.epsilon, cfg.lam)
    project_fn = lambda Zc: jnp.clip(Zc, 0.0, 1.0)
    return adam_scan(grad_fn, Z0, n_steps, cfg.code_lr, project_fn)


def infer_ista(X_new, D, cfg: ISTAConfig, n_steps):
    n, d_z = X_new.shape[0], D.shape[0]
    Z0 = jnp.zeros((n, d_z), dtype=X_new.dtype)
    grad_fn = lambda Zc: _recon_grad_batch(Zc, X_new, D)
    if cfg.box_constrained:
        project_fn = lambda Zc: jnp.clip(soft_threshold(Zc, cfg.alpha), 0.0, 1.0)
    else:
        project_fn = lambda Zc: soft_threshold(Zc, cfg.alpha)
    return adam_scan(grad_fn, Z0, n_steps, cfg.code_lr, project_fn)


# ============================================================
# 6. Metrics
# ============================================================

def support(Z, threshold=1e-2):
    return jnp.abs(Z) > threshold


def sparsity_stats(Z, threshold=1e-2):
    supp = support(Z, threshold)
    per_unit_active_freq = jnp.mean(supp, axis=0)
    return dict(
        mean_active_units=float(jnp.mean(jnp.sum(supp, axis=1))),
        per_unit_activity=np.asarray(jnp.mean(Z, axis=0)),
        per_unit_active_freq=np.asarray(per_unit_active_freq),
        dead_fraction=float(jnp.mean(per_unit_active_freq < 1e-4)),
    )


def binarity_score(Z, threshold=1e-2):
    """Mean of z*(1-z) over active units (0 = perfectly binary). Only
    meaningful for box-constrained codes in [0,1]."""
    supp = support(Z, threshold)
    penal = jnp.where(supp, Z * (1.0 - Z), 0.0)
    return float(jnp.sum(penal) / jnp.maximum(jnp.sum(supp), 1))


def support_f1(Z_pred, Z_true, threshold=1e-2):
    pred, true = support(Z_pred, threshold), support(Z_true, threshold)
    tp = jnp.sum(pred & true)
    fp = jnp.sum(pred & (~true))
    fn = jnp.sum((~pred) & true)
    precision = tp / jnp.maximum(tp + fp, 1)
    recall = tp / jnp.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / jnp.maximum(precision + recall, 1e-12)
    return float(f1), float(precision), float(recall)


def topk_support_f1(Z_pred, Z_true, k, threshold=1e-2):
    """Threshold-independent variant of support recovery: predicted support
    is the top-k (by magnitude) units per sample, vs. the thresholded true
    support. Avoids threshold/scale-sensitivity when comparing methods whose
    codes naturally live on different scales (e.g. unconstrained ISTA)."""
    d_z = Z_pred.shape[1]
    k = min(k, d_z)
    order = jnp.argsort(-jnp.abs(Z_pred), axis=1)[:, :k]
    pred = jnp.zeros_like(Z_pred, dtype=bool).at[jnp.arange(Z_pred.shape[0])[:, None], order].set(True)
    true = support(Z_true, threshold)
    tp = jnp.sum(pred & true)
    fp = jnp.sum(pred & (~true))
    fn = jnp.sum((~pred) & true)
    precision = tp / jnp.maximum(tp + fp, 1)
    recall = tp / jnp.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / jnp.maximum(precision + recall, 1e-12)
    return float(f1), float(precision), float(recall)


def dictionary_recovery(D_learned, D_true):
    """Best-matching cosine similarity between learned and true atoms via
    Hungarian assignment (resolves the permutation ambiguity; abs() accounts
    for the sign ambiguity of unconstrained dictionary learning)."""
    Dl, Dt = np.asarray(D_learned), np.asarray(D_true)
    Dl_n = Dl / (np.linalg.norm(Dl, axis=1, keepdims=True) + 1e-12)
    Dt_n = Dt / (np.linalg.norm(Dt, axis=1, keepdims=True) + 1e-12)
    sim = Dl_n @ Dt_n.T
    row_ind, col_ind = linear_sum_assignment(-np.abs(sim))
    matched_cos = np.abs(sim[row_ind, col_ind])
    norm_ratio = np.linalg.norm(Dl, axis=1)[row_ind] / (np.linalg.norm(Dt, axis=1)[col_ind] + 1e-12)
    return dict(mean_cosine=float(matched_cos.mean()), matched_cosine=matched_cos,
                norm_ratio=norm_ratio, assignment=(row_ind, col_ind))


def align_codes_to_ground_truth(Z_learned, dict_rec, d_z):
    """Permute the columns (units) of Z_learned into the ground-truth unit
    ordering, reusing the dictionary-recovery assignment. Required before
    comparing supports -- otherwise the comparison is against an essentially
    arbitrary permutation and is close to meaningless."""
    row_ind, col_ind = dict_rec["assignment"]
    perm = np.zeros(d_z, dtype=int)
    perm[col_ind] = row_ind  # perm[true_idx] = learned_idx
    return np.asarray(Z_learned)[:, perm]


def topk_reconstruction_curve(Z, X, D, k_values):
    """Reconstruction MSE keeping only the top-k (by magnitude) activations
    per sample -- the operationally relevant condition for hardware-efficient
    sparse inference (exact, hard sparsity, not just low expected activity)."""
    d_z = Z.shape[1]
    order = jnp.argsort(-jnp.abs(Z), axis=1)
    mses = []
    for k in k_values:
        k = min(k, d_z)
        idx = order[:, :k]
        mask = jnp.zeros_like(Z).at[jnp.arange(Z.shape[0])[:, None], idx].set(1.0)
        mses.append(float(jnp.mean((X - (Z * mask) @ D) ** 2)))
    return np.array(mses)


# ============================================================
# 7. Experiment runner
# ============================================================

def finalize_results(name, D, Z_train, Z_test, data, history):
    threshold = 1e-2
    train_stats = sparsity_stats(Z_train, threshold)
    test_stats = sparsity_stats(Z_test, threshold)

    dict_rec = dictionary_recovery(D, data["D_true"])
    Z_train_aligned = align_codes_to_ground_truth(Z_train, dict_rec, D.shape[0])
    f1, prec, rec = support_f1(jnp.asarray(Z_train_aligned), data["Z_train"], threshold)

    k_true = int(round(float(jnp.mean(jnp.sum(support(data["Z_train"], threshold), axis=1)))))
    f1k, preck, reck = topk_support_f1(jnp.asarray(Z_train_aligned), data["Z_train"], k_true, threshold)

    k_values = sorted(set([max(1, k_true // 2), max(1, k_true), k_true * 2, D.shape[0]]))
    topk_curve = topk_reconstruction_curve(Z_test, data["X_test"], D, k_values)

    return dict(
        name=name, D=np.asarray(D), history=history,
        train_mse=float(jnp.mean((data["X_train"] - Z_train @ D) ** 2)),
        test_mse=float(jnp.mean((data["X_test"] - Z_test @ D) ** 2)),
        train_stats=train_stats, test_stats=test_stats,
        support_f1=f1, support_precision=prec, support_recall=rec,
        support_f1_topk=f1k, support_precision_topk=preck, support_recall_topk=reck,
        k_true=k_true,
        dict_recovery=dict_rec,
        binarity=binarity_score(Z_train, threshold),
        binarity_test=binarity_score(Z_test, threshold),
        k_values=k_values, topk_curve=topk_curve,
    )


def run_ours(data, cfg: OursConfig, d_z, key):
    D, Z_train, history = train_ours(data["X_train"], cfg, d_z, key)
    z_bar_ref = jnp.mean(Z_train, axis=0)  # frozen reference statistic for inference
    Z_test = infer_ours(data["X_test"], D, z_bar_ref, cfg, cfg.code_inner_steps * 4)
    return finalize_results("ours", D, Z_train, Z_test, data, history)


def run_ista(data, cfg: ISTAConfig, d_z, key):
    D, Z_train, history = train_ista(data["X_train"], cfg, d_z, key)
    Z_test = infer_ista(data["X_test"], D, cfg, cfg.code_inner_steps * 4)
    tag = "l1_box" if cfg.box_constrained else "l1_free"
    return finalize_results(tag, D, Z_train, Z_test, data, history)


def run_single_config(data_cfg: DataConfig, ours_cfg: OursConfig, ista_cfg: ISTAConfig, seed):
    data = generate_dataset(data_cfg)
    k_ours, k_l1box, k_l1free = random.split(random.PRNGKey(seed), 3)
    results = {
        "ours": run_ours(data, ours_cfg, data_cfg.d_z, k_ours),
        "l1_box": run_ista(data, replace(ista_cfg, box_constrained=True), data_cfg.d_z, k_l1box),
        "l1_free": run_ista(data, replace(ista_cfg, box_constrained=False), data_cfg.d_z, k_l1free),
    }
    return results


# ============================================================
# 8. Sweeps
# ============================================================

def _sweep(param_name, values, set_value, data_cfg, ours_cfg, ista_cfg, seed):
    records = []
    for v in values:
        cfg = set_value(data_cfg, v)
        results = run_single_config(cfg, ours_cfg, ista_cfg, seed)
        for method, res in results.items():
            records.append(dict(
                sweep_param=param_name, value=v, method=method,
                test_mse=res["test_mse"], dict_cosine=res["dict_recovery"]["mean_cosine"],
                support_f1=res["support_f1"], support_f1_topk=res["support_f1_topk"],
                dead_fraction=res["train_stats"]["dead_fraction"],
                binarity=res["binarity"], mean_active=res["train_stats"]["mean_active_units"],
            ))
        print(f"  [{param_name}={v}] " +
              ", ".join(f"{m}: test_mse={r['test_mse']:.4f} active={r['train_stats']['mean_active_units']:.1f}"
                        for m, r in results.items()))
    return records


def sweep_sparsity(mu_values, data_cfg, ours_cfg, ista_cfg, seed=1):
    return _sweep("sparsity_mean", mu_values,
                   lambda c, v: replace(c, sparsity_mean=v), data_cfg, ours_cfg, ista_cfg, seed)


def sweep_coherence(coh_values, data_cfg, ours_cfg, ista_cfg, seed=2):
    return _sweep("coherence", coh_values,
                   lambda c, v: replace(c, coherence=v), data_cfg, ours_cfg, ista_cfg, seed)


def sweep_ours_lambda(lam_values, data_cfg, ours_cfg, seed=3):
    """Sweep the contrastive weight lambda for 'ours' on a single fixed
    dataset (so the comparison isolates the hyperparameter's effect)."""
    records = []
    data = generate_dataset(data_cfg)
    for lam in lam_values:
        cfg = replace(ours_cfg, lam=lam)
        res = run_ours(data, cfg, data_cfg.d_z, random.PRNGKey(seed))
        records.append(dict(
            sweep_param="lambda_ours", value=lam, method="ours",
            test_mse=res["test_mse"], dict_cosine=res["dict_recovery"]["mean_cosine"],
            support_f1=res["support_f1"], support_f1_topk=res["support_f1_topk"],
            dead_fraction=res["train_stats"]["dead_fraction"],
            binarity=res["binarity"], mean_active=res["train_stats"]["mean_active_units"],
        ))
        print(f"  [lambda={lam}] test_mse={res['test_mse']:.4f} "
              f"active={res['train_stats']['mean_active_units']:.1f} "
              f"dict_cos={res['dict_recovery']['mean_cosine']:.3f} "
              f"supp_f1={res['support_f1']:.3f} topk_f1={res['support_f1_topk']:.3f}")
    return records, data


def sweep_ista_alpha(alpha_values, data_cfg, ista_cfg, seed=4):
    """Sweep the L1 weight alpha for both ISTA variants on a single fixed
    dataset (reused, not regenerated, per alpha value)."""
    records = []
    data = generate_dataset(data_cfg)
    for alpha in alpha_values:
        for box in (True, False):
            cfg = replace(ista_cfg, alpha=alpha, box_constrained=box)
            key = random.PRNGKey(seed + (0 if box else 1))
            res = run_ista(data, cfg, data_cfg.d_z, key)
            records.append(dict(
                sweep_param="alpha_l1", value=alpha, method=res["name"],
                test_mse=res["test_mse"], dict_cosine=res["dict_recovery"]["mean_cosine"],
                support_f1=res["support_f1"], support_f1_topk=res["support_f1_topk"],
                dead_fraction=res["train_stats"]["dead_fraction"],
                binarity=res["binarity"], mean_active=res["train_stats"]["mean_active_units"],
            ))
            print(f"  [alpha={alpha}, box={box}] test_mse={res['test_mse']:.4f} "
                  f"active={res['train_stats']['mean_active_units']:.1f} "
                  f"dict_cos={res['dict_recovery']['mean_cosine']:.3f}")
    return records, data


# ============================================================
# 9. Plotting
# ============================================================

def plot_training_curves(results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, res in results.items():
        h = res["history"]
        axes[0].plot(h["mse"], label=name)
        axes[1].plot(h["active"], label=name)
        axes[2].plot(h["binarity"], label=name)
    axes[0].set_title("train reconstruction MSE"); axes[0].set_yscale("log")
    axes[1].set_title("mean active units / sample")
    axes[2].set_title("binarity score (active units)")
    for ax in axes:
        ax.set_xlabel("outer iteration")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_unit_activity(results, out_dir):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        res = results[name]
        ax.hist(res["train_stats"]["per_unit_active_freq"], bins=30)
        ax.set_title(f"{name}\ndead frac={res['train_stats']['dead_fraction']:.2f}")
        ax.set_xlabel("per-unit active frequency")
    axes[0].set_ylabel("# units")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_unit_activity.png"), dpi=150)
    plt.close()


def plot_topk_curves(results, out_dir):
    plt.figure(figsize=(6, 4))
    for name, res in results.items():
        plt.plot(res["k_values"], res["topk_curve"], marker="o", label=name)
    plt.xlabel("k (non-zero units kept)"); plt.ylabel("test reconstruction MSE"); plt.yscale("log")
    plt.legend(); plt.title("Hard top-k sparsification robustness"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "topk_curve.png"), dpi=150)
    plt.close()


def plot_sweep(records, metric, ylabel, out_path_template, log_y=False, ref_line=None, ref_label="ground truth"):
    by_param = collections.defaultdict(list)
    for r in records:
        by_param[r["sweep_param"]].append(r)
    for param_name, recs in by_param.items():
        methods = sorted(set(r["method"] for r in recs))
        plt.figure(figsize=(6, 4))
        for m in methods:
            xs = sorted(set(r["value"] for r in recs if r["method"] == m))
            ys = [np.mean([r[metric] for r in recs if r["method"] == m and r["value"] == x]) for x in xs]
            plt.plot(xs, ys, marker="o", label=m)
        if ref_line is not None:
            plt.axhline(ref_line, color="k", linestyle="--", alpha=0.6, label=ref_label)
        plt.xlabel(param_name); plt.ylabel(ylabel)
        if log_y:
            plt.yscale("log")
        plt.legend(); plt.title(f"{ylabel} vs {param_name}"); plt.tight_layout()
        plt.savefig(out_path_template.format(param=param_name), dpi=150)
        plt.close()


def print_summary(results):
    header = (f"{'method':10s} {'train_mse':>10s} {'test_mse':>10s} {'active/spl':>11s} "
              f"{'dead_frac':>10s} {'binarity':>9s} {'supp_F1':>8s} {'topk_F1':>8s} {'dict_cos':>9s}")
    print(header); print("-" * len(header))
    for name, res in results.items():
        print(f"{name:10s} {res['train_mse']:10.4f} {res['test_mse']:10.4f} "
              f"{res['train_stats']['mean_active_units']:11.2f} "
              f"{res['train_stats']['dead_fraction']:10.3f} "
              f"{res['binarity']:9.4f} {res['support_f1']:8.3f} {res['support_f1_topk']:8.3f} "
              f"{res['dict_recovery']['mean_cosine']:9.3f}")


# ============================================================
# 10. Gradient self-test (finite differences)
# ============================================================

def selftest_gradients(seed=0, d_z=12, d_x=8):
    key = random.PRNGKey(seed)
    k1, k2, k3, k4 = random.split(key, 4)
    z = random.uniform(k1, (d_z,))
    x = random.normal(k2, (d_x,))
    D = random.normal(k3, (d_z, d_x)) * 0.1
    z_bar = random.uniform(k4, (d_z,))
    epsilon, lam = 1e-2, 0.1

    analytic = np.asarray(_ours_grad_single(z, x, D, z_bar, epsilon, lam))
    eps_fd = 1e-6
    z_np = np.asarray(z)
    numeric = np.zeros(d_z)
    for i in range(d_z):
        zp, zm = z_np.copy(), z_np.copy()
        zp[i] += eps_fd; zm[i] -= eps_fd
        fp = float(ours_code_objective(jnp.array(zp), x, D, z_bar, epsilon, lam))
        fm = float(ours_code_objective(jnp.array(zm), x, D, z_bar, epsilon, lam))
        numeric[i] = (fp - fm) / (2 * eps_fd)

    err = np.max(np.abs(numeric - analytic))
    print(f"[selftest] max |analytic - finite-diff| gradient error (per-sample, frozen z_bar) = {err:.2e}")
    assert err < 1e-4, "Gradient self-test failed!"
    print("[selftest] per-sample gradient check passed.")


def selftest_mean_field_correction(seed=1, n=6, d_z=5):
    """Guards against the frozen-z_bar training bug regressing: verifies that
    (joint gradient) - (per-sample gradient with z_bar frozen) is a single
    constant vector broadcast across every sample -- the exact mean-field
    correction derived from differentiating z_bar = mean(Z) through the
    batch objective. If this ever fails, ours_code_step is once again
    silently dropping that correction."""
    key = random.PRNGKey(seed)
    Z = 0.3 * random.uniform(key, (n, d_z))
    epsilon, lam = 1e-2, 1.0

    def contrastive_only_batch(Z):
        z_bar = jnp.mean(Z, axis=0)
        u = jnp.sum(Z * Z, axis=1) + epsilon
        v = Z @ z_bar + epsilon
        return jnp.mean(lam * (-jnp.log(u) + jnp.log(v)))

    grad_joint = np.asarray(jax.grad(contrastive_only_batch)(Z))

    z_bar_fixed = jnp.mean(Z, axis=0)

    def contrastive_only_persample(z):
        u = jnp.dot(z, z) + epsilon
        v = jnp.dot(z, z_bar_fixed) + epsilon
        return lam * (-jnp.log(u) + jnp.log(v)) / n

    grad_frozen = np.asarray(jax.vmap(jax.grad(contrastive_only_persample))(Z))

    diff = grad_joint - grad_frozen
    row_std = np.std(diff, axis=0).max()
    print(f"[selftest] mean-field correction: max row-wise std of (joint - frozen) gradient = {row_std:.2e} "
          f"(should be ~0: the correction must be a single shared constant across samples)")
    assert row_std < 1e-6, "Mean-field correction is NOT a shared constant -- training gradient is likely wrong!"
    print("[selftest] mean-field correction structure confirmed.")


# ============================================================
# 11. Main
# ============================================================

def main(quick: bool = False, out_dir: str = "sbdr_dict_learning_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    selftest_gradients()
    selftest_mean_field_correction()

    if quick:
        data_cfg = DataConfig(d_x=24, d_z=32, n_groups=40, m_per_group=5,
                               n_test_groups=10, sparsity_mean=0.15)
        ours_cfg = OursConfig(outer_steps=8, code_inner_steps=8)
        ista_cfg = ISTAConfig(outer_steps=8, code_inner_steps=8)
        mu_values, coh_values = [0.1, 0.2], [0.0, 0.6]
        lam_values, alpha_values = [0.05, 0.3], [0.01, 0.1]
    else:
        data_cfg = DataConfig()
        ours_cfg = OursConfig(lam=3.0)
        ista_cfg = ISTAConfig()
        mu_values, coh_values = [0.05, 0.10, 0.20], [0.0, 0.5, 0.85]
        lam_values = [0.02, 0.05, 0.1, 0.3, 0.6, 1.0]
        alpha_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    print("\n=== Main experiment (default configuration) ===")
    t0 = time.time()
    results = run_single_config(data_cfg, ours_cfg, ista_cfg, seed=0)
    print(f"(finished in {time.time() - t0:.1f}s)\n")
    print_summary(results)
    true_active = sparsity_stats(generate_dataset(data_cfg)["Z_train"])["mean_active_units"]
    print(f"[diagnostic] ground-truth mean active units/sample = {true_active:.2f} "
          f"(compare to 'active/spl' column above)")
    print(f"[diagnostic] ours: mean self-similarity u=<z,z>+eps at end of training = "
          f"{results['ours']['history']['mean_u'][-1]:.3f} (epsilon={ours_cfg.epsilon}; "
          f"large u => weak contrastive restoring force, see module docstring)")
    plot_training_curves(results, out_dir)
    plot_unit_activity(results, out_dir)
    plot_topk_curves(results, out_dir)

    exit()

    print("\n=== Sweep: ground-truth sparsity level ===")
    sparsity_records = sweep_sparsity(mu_values, data_cfg, ours_cfg, ista_cfg)
    plot_sweep(sparsity_records, "test_mse", "test reconstruction MSE",
               os.path.join(out_dir, "sweep_{param}_test_mse.png"), log_y=True)
    plot_sweep(sparsity_records, "dict_cosine", "dictionary recovery (cosine)",
               os.path.join(out_dir, "sweep_{param}_dict_cosine.png"))
    plot_sweep(sparsity_records, "dead_fraction", "dead-unit fraction",
               os.path.join(out_dir, "sweep_{param}_dead_fraction.png"))

    print("\n=== Sweep: dictionary coherence ===")
    coherence_records = sweep_coherence(coh_values, data_cfg, ours_cfg, ista_cfg)
    plot_sweep(coherence_records, "test_mse", "test reconstruction MSE",
               os.path.join(out_dir, "sweep_{param}_test_mse.png"), log_y=True)
    plot_sweep(coherence_records, "dict_cosine", "dictionary recovery (cosine)",
               os.path.join(out_dir, "sweep_{param}_dict_cosine.png"))
    plot_sweep(coherence_records, "support_f1_topk", "top-k support recovery F1",
               os.path.join(out_dir, "sweep_{param}_support_f1_topk.png"))

    print("\n=== Sweep: contrastive weight lambda (ours) ===")
    lambda_records, lam_data = sweep_ours_lambda(lam_values, data_cfg, ours_cfg)
    true_active_lam = sparsity_stats(lam_data["Z_train"])["mean_active_units"]
    plot_sweep(lambda_records, "mean_active", "mean active units / sample",
               os.path.join(out_dir, "sweep_{param}_active.png"),
               ref_line=true_active_lam, ref_label="ground truth")
    plot_sweep(lambda_records, "dict_cosine", "dictionary recovery (cosine)",
               os.path.join(out_dir, "sweep_{param}_dict_cosine.png"))
    plot_sweep(lambda_records, "support_f1_topk", "top-k support recovery F1",
               os.path.join(out_dir, "sweep_{param}_support_f1_topk.png"))
    plot_sweep(lambda_records, "test_mse", "test reconstruction MSE",
               os.path.join(out_dir, "sweep_{param}_test_mse.png"), log_y=True)

    print("\n=== Sweep: L1 weight alpha (baseline, box & free) ===")
    alpha_records, alpha_data = sweep_ista_alpha(alpha_values, data_cfg, ista_cfg)
    true_active_alpha = sparsity_stats(alpha_data["Z_train"])["mean_active_units"]
    plot_sweep(alpha_records, "mean_active", "mean active units / sample",
               os.path.join(out_dir, "sweep_{param}_active.png"),
               ref_line=true_active_alpha, ref_label="ground truth")
    plot_sweep(alpha_records, "dict_cosine", "dictionary recovery (cosine)",
               os.path.join(out_dir, "sweep_{param}_dict_cosine.png"))
    plot_sweep(alpha_records, "support_f1_topk", "top-k support recovery F1",
               os.path.join(out_dir, "sweep_{param}_support_f1_topk.png"))

    print(f"\nAll figures saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    parser.add_argument("--out_dir", type=str, default="sbdr_dict_learning_outputs")
    args = parser.parse_args()
    main(quick=args.quick, out_dir=args.out_dir)