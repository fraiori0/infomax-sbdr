"""
Online Sparse-Sequence Retrieval Testbed  --  first slice
==========================================================

Modules implemented here:
  1. Synthetic generation        (motif dictionary, clean sequences, corruption)
     - including an ORDER-SENSITIVE regime where sequences share a vocabulary
       and differ only in temporal arrangement (content alone can't separate)
  2. Stored-sequence preprocessing (inverted index + prefix counts)
  3. Online tracker (the core graph/snap algorithm)
     - cost computation is a SWAPPABLE strategy (CostStrategy) so alternative
       residual->cost mappings (e.g. "features skipped in the middle") drop in
  4. Harness + evaluation (jitter sweep AND time-warp sweep)

Design defaults (agreed):
  - dense bool [T, D] arrays for generation / ground-truth
  - active-set lists (list[np.ndarray]) in the tracker hot path
  - track ALL stored sequences every sample (fully observable)
  - primary metric = predictive F1 over time; summary = classification accuracy

Pure NumPy / SciPy / pandas. Run:  python sparse_retrieval.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ============================================================================
# Config dataclasses
# ============================================================================

@dataclass
class SeqConfig:
    D: int = 128            # number of features (vocabulary size)
    n_active: int = 10       # target sparsity per frame (a)
    length: int = 100       # sequence length T (frames)
    n_motifs: int = 50      # size of motif dictionary
    motif_len: int = 5      # frames a motif spans
    motif_n_feat: int = 3   # features activated per motif frame
    sustain_prob: float = 0.3   # prob a feature carries to next frame
    seed: int = 0
    # --- order-sensitive regime ---
    # When True, every sequence is built from the SAME small set of motifs in a
    # DIFFERENT order. Feature content (histogram) is then ~identical across
    # sequences, so only temporal ORDER discriminates them. This is the regime
    # where a bag-of-features baseline should collapse to chance.
    order_sensitive: bool = False
    n_motifs_per_seq: int = 8   # how many motif-slots make up one sequence
    shared_pool_size: int = 30   # motifs shared across all sequences (order mode)


@dataclass
class CorruptConfig:
    jitter_std: float = 0.0     # temporal jitter (frames, gaussian)
    delay: int = 0              # global offset (frames)
    p_drop: float = 0.0         # per-activation deletion prob
    p_add: float = 0.0          # spurious activation prob (per feature-slot)
    warp_amp: float = 0.0       # sinusoidal time-warp amplitude (frames)
    warp_period: float = 80.0   # sinusoidal warp period (frames)
    seed: int = 0


@dataclass
class TrackerConfig:
    W: float = 8.0          # gating window (half-width) for nearest search
    Delta_max: float = 12.0 # cost assigned to a miss
    lam: float = 0.9        # leak for dissimilarity / miss integrators
    beta: float = 0.2       # velocity smoothing
    alpha: float = 2.0      # backward-residual penalty (soft monotonicity)
    vel_lo: float = 0.5     # velocity clamp range
    vel_hi: float = 2.0
    consensus: str = "median"   # "median" | "mean"
    cost: str = "abs"       # cost strategy: "abs" | "skip"  (see COST_STRATEGIES)


# ============================================================================
# 1. Synthetic generation
# ============================================================================

def make_motif_dictionary(cfg: SeqConfig, rng: np.random.Generator):
    """Each motif: a (motif_len, motif_n_feat) array of feature indices.

    Think of a motif as a short 'note/phoneme': a fixed little pattern of
    feature activations spanning `motif_len` frames.
    """
    motifs = []
    for _ in range(cfg.n_motifs):
        # features the motif draws from (a small pool, reused across its frames
        # so motifs have internal feature coherence -> recurring features)
        pool = rng.choice(cfg.D, size=min(cfg.motif_n_feat * 2, cfg.D),
                          replace=False)
        pattern = np.stack([
            rng.choice(pool, size=cfg.motif_n_feat, replace=False)
            for _ in range(cfg.motif_len)
        ])
        motifs.append(pattern)
    return motifs


def generate_clean_sequence(cfg: SeqConfig, motifs, rng: np.random.Generator):
    """Schedule motifs over time, union their features per frame, add sustain.

    Returns dense bool [T, D].
    """
    seq = np.zeros((cfg.length, cfg.D), dtype=bool)
    t = 0
    while t < cfg.length:
        m = motifs[rng.integers(len(motifs))]
        for i in range(m.shape[0]):
            if t + i >= cfg.length:
                break
            seq[t + i, m[i]] = True
        t += m.shape[0]
        # occasional gap between motifs
        t += int(rng.integers(0, 2))

    return _finalize_sequence(seq, cfg, rng)


def generate_ordered_sequence(cfg: SeqConfig, shared_motifs, order,
                              rng: np.random.Generator):
    """Build a sequence as a fixed ORDER of shared motifs (order-sensitive mode).

    `order` is a sequence of indices into `shared_motifs`. The order is TILED
    (repeated) as needed to fill cfg.length, so there is no large empty tail.
    Two sequences with the same multiset but different `order` have ~identical
    feature histograms yet are temporally distinct -> only order discriminates.
    """
    seq = np.zeros((cfg.length, cfg.D), dtype=bool)
    t = 0
    oi = 0
    order = list(order)
    while t < cfg.length:
        m = shared_motifs[order[oi % len(order)]]
        oi += 1
        for i in range(m.shape[0]):
            if t + i >= cfg.length:
                break
            seq[t + i, m[i]] = True
        t += m.shape[0]
    return _finalize_sequence(seq, cfg, rng)


def _finalize_sequence(seq, cfg: SeqConfig, rng: np.random.Generator):
    """Shared post-processing: sustain + sparsity cap."""
    T = seq.shape[0]
    for t in range(1, T):
        carry = seq[t - 1] & (rng.random(cfg.D) < cfg.sustain_prob)
        seq[t] |= carry
    for t in range(T):
        idx = np.flatnonzero(seq[t])
        if len(idx) > cfg.n_active:
            keep = rng.choice(idx, size=cfg.n_active, replace=False)
            seq[t] = False
            seq[t, keep] = True
    return seq


def build_database(seq_cfg: SeqConfig, n_sequences: int):
    """Construct a list of stored sequences under either regime.

    Returns (stored_seqs, motifs). In order-sensitive mode all sequences share
    one motif pool AND the same motif multiset; they differ ONLY by a permuted
    order. Motifs are tiled to fill the full sequence length so there are no
    large empty tails (which would wash out the discriminative early span).
    """
    rng = np.random.default_rng(seq_cfg.seed)
    if seq_cfg.order_sensitive:
        pool = make_motif_dictionary(
            SeqConfig(**{**seq_cfg.__dict__,
                         "n_motifs": seq_cfg.shared_pool_size}), rng)
        # Size the multiset to roughly fill `length` with ONE non-repeating
        # permutation (tiling a short order would make permutations cyclically
        # equivalent and weaken order-discrimination). ~length/motif_len slots.
        n_slots = max(seq_cfg.n_motifs_per_seq,
                      int(np.ceil(seq_cfg.length / seq_cfg.motif_len)))
        # ONE fixed multiset of motif slots, shared by every sequence.
        # Each sequence is a distinct PERMUTATION of this same multiset, so
        # feature histograms are ~identical and only ORDER discriminates.
        base_slots = rng.integers(0, seq_cfg.shared_pool_size, size=n_slots)
        stored = []
        for i in range(n_sequences):
            r = np.random.default_rng(seq_cfg.seed + i + 1)
            order = r.permutation(base_slots)
            stored.append(generate_ordered_sequence(seq_cfg, pool, order, r))
        return stored, pool
    else:
        motifs = make_motif_dictionary(seq_cfg, rng)
        stored = []
        for i in range(n_sequences):
            c = SeqConfig(**{**seq_cfg.__dict__, "seed": seq_cfg.seed + i + 1})
            r = np.random.default_rng(c.seed)
            stored.append(generate_clean_sequence(c, motifs, r))
        return stored, motifs


def _warp_map(T: int, cfg: CorruptConfig):
    """Map query-time -> source-time via delay + sinusoidal warp.

    Returns float array `src[t]` = position in the clean sequence that query
    frame t samples from. Also the natural ground-truth for alignment eval.
    """
    t = np.arange(T)
    src = t - cfg.delay
    if cfg.warp_amp != 0.0:
        src = src + cfg.warp_amp * np.sin(2 * np.pi * t / cfg.warp_period)
    return src


def corrupt_sequence(clean: np.ndarray, cfg: CorruptConfig):
    """Apply delay/warp/jitter/dropout/insertion. Returns (query_bool, truth).

    truth = dict(src=float[T])  the source position each query frame maps to.
    """
    rng = np.random.default_rng(cfg.seed)
    T, D = clean.shape
    src = _warp_map(T, cfg)
    query = np.zeros_like(clean)

    for t in range(T):
        s = src[t]
        if cfg.jitter_std > 0:
            s = s + rng.normal(0, cfg.jitter_std)
        si = int(round(s))
        if 0 <= si < T:
            active = np.flatnonzero(clean[si])
            # dropout
            if cfg.p_drop > 0 and len(active):
                active = active[rng.random(len(active)) >= cfg.p_drop]
            query[t, active] = True
        # insertion noise
        if cfg.p_add > 0:
            add = np.flatnonzero(rng.random(D) < cfg.p_add)
            query[t, add] = True

    return query, {"src": src}


def to_active_sets(seq: np.ndarray):
    """Dense bool [T,D] -> list of sorted int arrays (active-set hot format)."""
    return [np.flatnonzero(seq[t]) for t in range(seq.shape[0])]


# ============================================================================
# 2. Preprocessing: inverted index + prefix counts
# ============================================================================

def build_index(seq: np.ndarray):
    """inverted[d] = sorted np.array of times feature d is active.
    prefix[t] = number of activations in frames [0, t)  (for expected-count).
    """
    T, D = seq.shape
    inverted = [np.flatnonzero(seq[:, d]) for d in range(D)]
    counts = seq.sum(axis=1)
    prefix = np.concatenate([[0], np.cumsum(counts)])  # len T+1
    return {"inverted": inverted, "prefix": prefix, "T": T, "D": D,
            "active_sets": to_active_sets(seq)}


def _nearest_in_window(times: np.ndarray, center: float, W: float):
    """Nearest element of sorted `times` to `center`, within +-W. None if empty.
    O(log n) via searchsorted.
    """
    if times.size == 0:
        return None
    lo = np.searchsorted(times, center - W, side="left")
    hi = np.searchsorted(times, center + W, side="right")
    if lo >= hi:
        return None
    cand = times[lo:hi]
    j = np.argmin(np.abs(cand - center))
    return float(cand[j])


# ============================================================================
# 3. Online tracker (core algorithm)
# ============================================================================

# --- Cost strategies (swappable) -------------------------------------------
# A cost strategy maps a matched residual into an edge cost. It receives the
# full context so richer strategies (e.g. counting activations skipped between
# the predicted position and the matched one) can be implemented without
# touching the tracker. Signature:
#     cost(residual, predicted_pos, matched_pos, index_dict, cfg) -> float
# Return a non-negative scalar; the tracker clips to Delta_max itself.

def cost_asymmetric_abs(r, p, t_d, idx, cfg):
    """Default: |r|, with backward residuals penalized by alpha."""
    c = abs(r)
    if r < 0:
        c *= cfg.alpha
    return c


def cost_skip_count(r, p, t_d, idx, cfg):
    """Skip-based cost (your idea): how many stored activations lie BETWEEN the
    predicted position and the matched one. A big temporal jump that also skips
    a lot of content is costlier than a jump across a quiet gap.

    Uses the prefix-count table for an O(1) activation count in [lo, hi).
    Normalised by frame count so it stays comparable to the abs-residual scale,
    then blended with a small positional term so ties break sensibly.
    """
    prefix = idx["prefix"]
    lo, hi = sorted((int(round(p)), int(round(t_d))))
    lo = max(0, min(lo, len(prefix) - 1))
    hi = max(0, min(hi, len(prefix) - 1))
    skipped_activations = prefix[hi] - prefix[lo]           # O(1)
    span = max(hi - lo, 1)
    # per-frame density of skipped content; backward still penalised
    c = skipped_activations / span + 0.25 * abs(r)
    if r < 0:
        c *= cfg.alpha
    return c


COST_STRATEGIES = {
    "abs": cost_asymmetric_abs,
    "skip": cost_skip_count,
}


class OnlineTracker:
    """Tracks phase/velocity/dissimilarity for every stored sequence at once."""

    def __init__(self, stored_indices, cfg: TrackerConfig):
        self.idx = stored_indices            # list of build_index() dicts
        self.cfg = cfg
        self.K = len(stored_indices)
        self.cost_fn = COST_STRATEGIES[getattr(cfg, "cost", "abs")]
        self.reset()

    def reset(self):
        self.phase = np.zeros(self.K)        # τ̂_k
        self.vel = np.ones(self.K)           # v̂_k
        self.dissim = np.zeros(self.K)       # D_k  (leaky)
        self.miss = np.zeros(self.K)         # m_k  (leaky)
        self._t = 0

    def _edge_cost(self, r, p, t_d, k) -> float:
        """Apply the configured strategy, then clip to Delta_max."""
        c = self.cost_fn(r, p, t_d, self.idx[k], self.cfg)
        return min(c, self.cfg.Delta_max)

    def step(self, q_active: np.ndarray, record=False):
        """Consume one query active-set; update all sequences' state.

        If record=True, also return per-sequence step internals (median
        residual, match fraction, step cost) for visualization. This adds
        a few arrays per call, so leave it off in the hot evaluation path.
        """
        cfg = self.cfg
        rec_tilde = np.zeros(self.K) if record else None
        rec_matchfrac = np.zeros(self.K) if record else None
        rec_stepcost = np.zeros(self.K) if record else None
        for k in range(self.K):
            inv = self.idx[k]["inverted"]
            p = self.phase[k] + self.vel[k]          # predicted position
            residuals = []
            costs = []
            n_miss = 0
            for d in q_active:
                if d >= len(inv):
                    n_miss += 1
                    continue
                t_d = _nearest_in_window(inv[d], p, cfg.W)
                if t_d is None:
                    n_miss += 1
                else:
                    residuals.append(t_d - p)
                    costs.append(self._edge_cost(t_d - p, p, t_d, k))

            n_q = max(len(q_active), 1)
            if residuals:
                r = np.asarray(residuals)
                tilde = np.median(r) if cfg.consensus == "median" else r.mean()
                step_cost = float(np.mean(costs))
                miss_frac = n_miss / n_q
                step_cost = step_cost * (1 - miss_frac) + cfg.Delta_max * miss_frac
            else:
                tilde = 0.0
                step_cost = cfg.Delta_max
                miss_frac = 1.0

            # state updates
            self.phase[k] = p + tilde
            new_v = (1 - cfg.beta) * self.vel[k] + cfg.beta * (1.0 + tilde)
            self.vel[k] = np.clip(new_v, cfg.vel_lo, cfg.vel_hi)
            self.dissim[k] = cfg.lam * self.dissim[k] + (1 - cfg.lam) * step_cost
            self.miss[k] = cfg.lam * self.miss[k] + (1 - cfg.lam) * miss_frac

            if record:
                rec_tilde[k] = tilde
                rec_matchfrac[k] = 1.0 - miss_frac
                rec_stepcost[k] = step_cost

        self._t += 1
        out = {"phase": self.phase.copy(), "vel": self.vel.copy(),
               "dissim": self.dissim.copy(), "miss": self.miss.copy()}
        if record:
            out.update({"tilde": rec_tilde, "match_frac": rec_matchfrac,
                        "step_cost": rec_stepcost})
        return out

    def predict_next(self, k: int) -> np.ndarray:
        """Predicted active set for the next frame of sequence k (for F1 eval)."""
        pos = int(round(self.phase[k] + self.vel[k]))
        sets = self.idx[k]["active_sets"]
        if 0 <= pos < len(sets):
            return sets[pos]
        return np.array([], dtype=int)

    def scores(self) -> np.ndarray:
        """Similarity = -dissimilarity (higher is better)."""
        return -self.dissim


# ============================================================================
# 4. Harness + evaluation
# ============================================================================

def f1_sets(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) == 0 and len(true) == 0:
        return 1.0
    if len(pred) == 0 or len(true) == 0:
        return 0.0
    inter = len(np.intersect1d(pred, true, assume_unique=False))
    prec = inter / len(pred)
    rec = inter / len(true)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def run_query(tracker: OnlineTracker, query: np.ndarray, true_k: int):
    """Stream a query through the tracker. Returns trace dict.

    Predictive F1: at each t we predict frame t+1's active set for the TRUE
    sequence, then compare to the actual next query frame.
    """
    tracker.reset()
    q_sets = to_active_sets(query)
    T = len(q_sets)
    pred_f1 = np.full(T, np.nan)
    phase_track = np.zeros(T)

    for t in range(T):
        # predict next BEFORE consuming, using current state
        pred = tracker.predict_next(true_k)
        if t + 1 < T:
            pred_f1[t] = f1_sets(pred, q_sets[t + 1])
        tracker.step(q_sets[t])
        phase_track[t] = tracker.phase[true_k]

    scores = tracker.scores()
    pred_k = int(np.argmax(scores))
    # margin: best vs second-best
    order = np.argsort(scores)[::-1]
    margin = scores[order[0]] - scores[order[1]] if len(scores) > 1 else 0.0

    return {"pred_k": pred_k, "correct": pred_k == true_k, "margin": margin,
            "pred_f1": pred_f1, "phase_track": phase_track,
            "scores": scores}


def bag_of_features_baseline(stored_seqs, query):
    """Time-agnostic baseline: cosine between summed feature histograms."""
    q = query.sum(axis=0).astype(float)
    qn = np.linalg.norm(q) + 1e-9
    best, best_k = -1.0, -1
    for k, s in enumerate(stored_seqs):
        v = s.sum(axis=0).astype(float)
        c = float(q @ v) / ((np.linalg.norm(v) + 1e-9) * qn)
        if c > best:
            best, best_k = c, k
    return best_k


def _sweep(seq_cfg, trk_cfg, make_corrupt, levels, level_name,
           n_sequences, n_trials, base_seed):
    """Generic corruption-sweep driver.

    make_corrupt(level, trial_seed) -> CorruptConfig
    Builds the DB once, sweeps `levels`, runs n_trials per level.
    """
    stored, _ = build_database(seq_cfg, n_sequences)
    indices = [build_index(s) for s in stored]
    tracker = OnlineTracker(indices, trk_cfg)

    rows = []
    for lev in levels:
        for trial in range(n_trials):
            true_k = (base_seed + trial) % n_sequences
            cc = make_corrupt(lev, base_seed * 1000 + trial)
            query, truth = corrupt_sequence(stored[true_k], cc)
            res = run_query(tracker, query, true_k)
            bof_k = bag_of_features_baseline(stored, query)
            al_err = np.nanmean(np.abs(res["phase_track"] - truth["src"]))
            rows.append({
                level_name: lev, "trial": trial,
                "correct": res["correct"],
                "bof_correct": bof_k == true_k,
                "margin": res["margin"],
                "pred_f1": np.nanmean(res["pred_f1"]),
                "align_err": al_err,
            })
    return pd.DataFrame(rows)


def jitter_sweep(seq_cfg, trk_cfg, n_sequences=20,
                 jitter_levels=(0, 1, 2, 4, 8), n_trials=10, base_seed=0):
    return _sweep(
        seq_cfg, trk_cfg,
        make_corrupt=lambda lev, s: CorruptConfig(jitter_std=float(lev), seed=s),
        levels=jitter_levels, level_name="jitter",
        n_sequences=n_sequences, n_trials=n_trials, base_seed=base_seed)


def warp_sweep(seq_cfg, trk_cfg, n_sequences=20,
               warp_levels=(0, 2, 4, 8, 12), warp_period=80.0,
               n_trials=10, base_seed=0):
    """Sinusoidal time-warp sweep -- the headline perturbation for the velocity
    tracker. warp_amp is the peak displacement (frames) of the warp."""
    return _sweep(
        seq_cfg, trk_cfg,
        make_corrupt=lambda lev, s: CorruptConfig(
            warp_amp=float(lev), warp_period=warp_period, seed=s),
        levels=warp_levels, level_name="warp_amp",
        n_sequences=n_sequences, n_trials=n_trials, base_seed=base_seed)


# ============================================================================
# main
# ============================================================================

def record_full_trace(tracker: OnlineTracker, query: np.ndarray, true_k: int):
    """Stream a query, recording every sequence's state at every step.

    Returns a dict of [T, K] arrays (phase, vel, dissim, miss, tilde,
    match_frac, step_cost) plus the per-step predictive F1 for true_k.
    Used by the visualization; keeps full history so we can plot evolution.
    """
    tracker.reset()
    q_sets = to_active_sets(query)
    T, K = len(q_sets), tracker.K
    hist = {key: np.zeros((T, K)) for key in
            ["phase", "vel", "dissim", "miss", "tilde", "match_frac", "step_cost"]}
    pred_f1 = np.full(T, np.nan)

    for t in range(T):
        pred = tracker.predict_next(true_k)
        if t + 1 < T:
            pred_f1[t] = f1_sets(pred, q_sets[t + 1])
        out = tracker.step(q_sets[t], record=True)
        for key in hist:
            hist[key][t] = out[key]
    hist["pred_f1"] = pred_f1
    hist["q_sets"] = q_sets
    return hist


# ============================================================================
# 5. Visualization (matplotlib)
# ============================================================================

def plot_motif_dictionary(motifs, D, path="viz_motifs.png", max_motifs=12):
    """Show each motif as a (motif_len x D) activation grid."""
    import matplotlib.pyplot as plt
    n = min(len(motifs), max_motifs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 1.6 * nrow),
                             squeeze=False)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i < n:
            m = motifs[i]                       # (motif_len, motif_n_feat)
            grid = np.zeros((m.shape[0], D))
            for f in range(m.shape[0]):
                grid[f, m[f]] = 1
            ax.imshow(grid.T, aspect="auto", cmap="Greys",
                      interpolation="nearest")
            ax.set_title(f"motif {i}", fontsize=8)
            ax.set_xlabel("frame", fontsize=7)
            if i % ncol == 0:
                ax.set_ylabel("feature", fontsize=7)
        else:
            ax.axis("off")
    fig.suptitle("Motif dictionary", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_example_sequences(stored, n=4, path="viz_sequences.png"):
    """Show a few stored sequences as feature-vs-time raster plots."""
    import matplotlib.pyplot as plt
    n = min(n, len(stored))
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n), squeeze=False)
    for i in range(n):
        ax = axes[i][0]
        ax.imshow(stored[i].T, aspect="auto", cmap="Greys",
                  interpolation="nearest")
        ax.set_ylabel(f"seq {i}\nfeature", fontsize=8)
        if i == n - 1:
            ax.set_xlabel("time (frames)")
    fig.suptitle("Example stored sequences (feature x time)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_online_behavior(hist, true_k, top_n=5, path="viz_online.png"):
    """The main diagnostic: how the tracker behaves as samples arrive.

    Panels (shared time axis):
      1. Similarity (-dissim) of the top-N sequences over time; true_k bold.
      2. Estimated phase of the top sequences vs the ideal diagonal.
      3. Velocity of true_k (warp rate) over time.
      4. Per-step median residual (tilde) and match fraction of true_k.
      5. Predictive F1 of true_k over time.
    """
    import matplotlib.pyplot as plt
    T, K = hist["dissim"].shape
    sim = -hist["dissim"]                         # [T, K]
    final_rank = np.argsort(sim[-1])[::-1]
    top = list(final_rank[:top_n])
    if true_k not in top:
        top = [true_k] + top[:top_n - 1]

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)

    # 1. similarity over time
    ax = axes[0]
    for k in top:
        lw, z = (2.5, 3) if k == true_k else (1.0, 1)
        label = f"seq {k}" + (" (true)" if k == true_k else "")
        ax.plot(sim[:, k], lw=lw, zorder=z, label=label)
    ax.set_ylabel("similarity\n(-dissim)")
    ax.legend(fontsize=7, ncol=top_n, loc="lower left")
    ax.set_title(f"Online tracker behavior  (true sequence = {true_k})")

    # 2. estimated phase vs ideal
    ax = axes[1]
    ax.plot(np.arange(T), "k--", lw=1, alpha=0.5, label="ideal (slope 1)")
    for k in top:
        lw = 2.5 if k == true_k else 1.0
        ax.plot(hist["phase"][:, k], lw=lw,
                label=f"seq {k}" + (" (true)" if k == true_k else ""))
    ax.set_ylabel("estimated\nphase")
    ax.legend(fontsize=7, loc="upper left")

    # 3. velocity of true_k
    ax = axes[2]
    ax.plot(hist["vel"][:, true_k], color="tab:green")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_ylabel("velocity\n(true_k)")

    # 4. per-step median residual + match fraction of true_k
    ax = axes[3]
    ax.plot(hist["tilde"][:, true_k], color="tab:blue", label="median residual")
    ax.axhline(0.0, color="k", ls=":", alpha=0.5)
    ax.set_ylabel("median resid")
    ax2 = ax.twinx()
    ax2.plot(hist["match_frac"][:, true_k], color="tab:orange", alpha=0.7,
             label="match fraction")
    ax2.set_ylabel("match frac", color="tab:orange")
    ax2.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")

    # 5. predictive F1
    ax = axes[4]
    ax.plot(hist["pred_f1"], color="tab:purple")
    ax.set_ylabel("predictive F1\n(true_k)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time (query frames)")

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_snapshot(hist, true_k, t, top_n=5, path="viz_snapshot.png"):
    """Freeze one timestep: bar chart of current top-N matched sequences with
    their similarity, plus a text box of true_k's live variables. Shows what
    the system 'believes' at frame t as if paused mid-stream."""
    import matplotlib.pyplot as plt
    sim = -hist["dissim"][t]                      # [K]
    order = np.argsort(sim)[::-1][:top_n]
    fig, (axb, axt) = plt.subplots(1, 2, figsize=(11, 3.2),
                                   gridspec_kw={"width_ratios": [2, 1]})
    colors = ["tab:red" if k == true_k else "tab:blue" for k in order]
    axb.barh([f"seq {k}" for k in order][::-1], sim[order][::-1],
             color=colors[::-1])
    axb.set_xlabel("similarity (-dissim)")
    axb.set_title(f"Top-{top_n} matches at frame t={t}")

    txt = (f"true_k = {true_k}\n"
           f"phase  = {hist['phase'][t, true_k]:.2f}\n"
           f"vel    = {hist['vel'][t, true_k]:.3f}\n"
           f"median resid = {hist['tilde'][t, true_k]:.2f}\n"
           f"match frac   = {hist['match_frac'][t, true_k]:.2f}\n"
           f"step cost    = {hist['step_cost'][t, true_k]:.2f}\n"
           f"dissim (leaky)= {hist['dissim'][t, true_k]:.3f}\n"
           f"pred F1      = {hist['pred_f1'][t]:.2f}")
    axt.axis("off")
    axt.text(0.0, 0.95, txt, va="top", ha="left", family="monospace",
             fontsize=10)
    axt.set_title("true_k live variables")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def diagnose_scores(seq_cfg, trk_cfg, n_sequences=20, true_k=3,
                    corrupt=None, verbose=True):
    """Inspect the full score vector for one query: are scores saturated?
    is the correct sequence actually separated? Returns a dict of stats.
    """
    stored, _ = build_database(seq_cfg, n_sequences)
    indices = [build_index(s) for s in stored]
    tracker = OnlineTracker(indices, trk_cfg)
    if corrupt is None:
        corrupt = CorruptConfig(jitter_std=0.0, seed=42)
    query, _ = corrupt_sequence(stored[true_k], corrupt)
    res = run_query(tracker, query, true_k)
    sc = res["scores"]
    order = np.argsort(sc)[::-1]
    diss = -sc
    stats = {
        "pred_k": res["pred_k"], "true_k": true_k,
        "correct": res["correct"],
        "n_distinct_scores": int(len(np.unique(np.round(sc, 6)))),
        "all_saturated": bool(np.allclose(diss, trk_cfg.Delta_max)),
        "true_rank": int(np.flatnonzero(order == true_k)[0]),
        "best_diss": float(diss.min()), "worst_diss": float(diss.max()),
        "margin": float(res["margin"]),
        "empty_frame_frac": float((query.sum(axis=1) == 0).mean()),
    }
    if verbose:
        print(f"  pred_k={stats['pred_k']} true_k={true_k} "
              f"correct={stats['correct']} true_rank={stats['true_rank']}")
        print(f"  distinct_scores={stats['n_distinct_scores']} "
              f"all_saturated={stats['all_saturated']} "
              f"empty_frames={stats['empty_frame_frac']:.2f}")
        print(f"  dissim: best={stats['best_diss']:.3f} "
              f"worst={stats['worst_diss']:.3f} margin={stats['margin']:.3f}")
        print("  top5 (k, dissim):",
              [(int(k), round(float(diss[k]), 3)) for k in order[:5]])
    return stats


def run_visualizations(order_sensitive=True, n_sequences=20, true_k=3,
                       corrupt=None, warp_amp=8.0, snapshot_t=None):
    """Generate the full visualization set for one example query."""
    seq_cfg = SeqConfig(order_sensitive=order_sensitive)
    trk_cfg = TrackerConfig(
        consensus="mean",
        cost="skip", #"skip",
    )
    stored, motifs = build_database(seq_cfg, n_sequences)
    indices = [build_index(s) for s in stored]
    tracker = OnlineTracker(indices, trk_cfg)

    if corrupt is None:
        corrupt = CorruptConfig(warp_amp=warp_amp, warp_period=80.0, seed=7)
    query, truth = corrupt_sequence(stored[true_k], corrupt)
    hist = record_full_trace(tracker, query, true_k)

    T = hist["dissim"].shape[0]
    if snapshot_t is None:
        snapshot_t = T // 2

    paths = []
    paths.append(plot_motif_dictionary(motifs, seq_cfg.D))
    paths.append(plot_example_sequences(stored))
    paths.append(plot_online_behavior(hist, true_k))
    paths.append(plot_snapshot(hist, true_k, snapshot_t))
    print("Saved visualizations:")
    for p in paths:
        print("  ", p)
    return paths


if __name__ == "__main__":
    import sys

    if "--viz" in sys.argv:
        run_visualizations(
            order_sensitive=True,
            n_sequences=500,
            true_k=7,
            # corrupt the sequence
            corrupt=CorruptConfig(
                jitter_std=1.0,
                seed=42,
                p_drop=0.02,
                p_add=0.02,
                delay=0,
            ),
            warp_amp=8.0,
            snapshot_t=None,
        )
        sys.exit(0)

    def show(df, level_name):
        summ = df.groupby(level_name).agg(
            acc=("correct", "mean"),
            bof_acc=("bof_correct", "mean"),
            pred_f1=("pred_f1", "mean"),
            margin=("margin", "mean"),
            align_err=("align_err", "mean"),
        ).round(3)
        print(summ.to_string())

    trk_cfg = TrackerConfig()

    # --- Score diagnostics (check saturation / margin) --------------------
    print("\n=== Score diagnostics: content-separable regime ===")
    diagnose_scores(SeqConfig(), trk_cfg)
    print("\n=== Score diagnostics: ORDER-sensitive regime ===")
    diagnose_scores(SeqConfig(order_sensitive=True), trk_cfg)

    # --- A. content-separable regime (sanity: baseline strong) -------------
    print("\n=== A. Jitter sweep, content-separable regime ===")
    print("(distinct motifs per sequence -> bag-of-features should stay high)\n")
    show(jitter_sweep(SeqConfig(), trk_cfg, n_sequences=300), "jitter")

    # --- B. ORDER-sensitive regime (the hard test) ------------------------
    print("\n=== B. Jitter sweep, ORDER-sensitive regime ===")
    print("(shared multiset, only order differs -> bof should collapse)\n")
    show(jitter_sweep(SeqConfig(order_sensitive=True), trk_cfg,
                      n_sequences=300), "jitter")

    # --- C. TIME-WARP sweep, order-sensitive regime -----------------------
    print("\n=== C. Time-warp sweep, ORDER-sensitive regime ===")
    print("(sinusoidal warp; exercises the velocity tracker)\n")
    show(warp_sweep(SeqConfig(order_sensitive=True), trk_cfg,
                    n_sequences=300), "warp_amp")

    print("\nColumns: acc=our top-1, bof_acc=baseline top-1,")
    print("pred_f1=predictive F1 (primary), margin=score gap, "
          "align_err=mean |phase - true src|")