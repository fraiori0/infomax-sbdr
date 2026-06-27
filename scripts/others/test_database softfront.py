"""
SBDR Sequence Retrieval via a Leaky Vote Accumulator  --  self-contained testbed
================================================================================

Single matching criterion (no weighted-sum tuning):

  Each active query feature votes for the offset(s) that would explain it.
  Votes accumulate into a per-sequence, leaky, soft-binned OFFSET HISTOGRAM.
  Everything is read off the shape of that one histogram:

      phase correction   = peak offset            argmax_delta H(delta)
      match quality      = peak PROMINENCE         (recall x consensus)
      silence            = no votes -> abstain (phase coasts)
      noise              = non-concentrating votes -> ignored by the peak

  Version implemented here (per the agreed plan):
    - ABSOLUTE-offset accumulator (Option A): warp tolerance via SOFT BINS only,
      no velocity-centering yet (that is the earned upgrade for later).
    - PER-SEQUENCE accumulator (clear to inspect; global index is a later
      refactor for the ~1000-sequence scaling regime).

Generation models a TCN->SBDR encoder, NOT audio:
    phones (overlapping sparse prototypes) -> classes (ordered phone strings,
    shared vocabulary) -> utterances (variable silence, duration jitter = warp,
    bit dropout/insertion = encoder noise, phone substitution = confusable).

Baselines: bag-of-features (floor), single-offset cross-correlation (ablation of
the temporal accumulator), windowed DTW (warp-optimal ceiling).

Pure NumPy / SciPy / pandas / matplotlib.
  python sbdr_voting.py          # run evaluation sweeps
  python sbdr_voting.py --viz    # generate visualizations
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ============================================================================
# Config
# ============================================================================

@dataclass
class GenConfig:
    D: int = 256                # SBDR dimensionality
    code_density: int = 8       # active bits per phone prototype (sparsity a)
    n_phones: int = 24          # size of shared phone bank
    phone_pool: int = 48        # feature pool phones draw from (controls overlap)
    n_classes: int = 20         # number of distinct "words"/classes
    phones_per_class: int = 6   # phones in a class string
    nominal_dur: int = 6        # nominal frames per phone
    seed: int = 0
    # order-sensitive regime: every class is a PERMUTATION of the SAME phone
    # multiset, so phone histograms are ~identical across classes and only the
    # temporal ARRANGEMENT discriminates (collapses bag-of-features to chance).
    order_sensitive: bool = False


@dataclass
class UttConfig:
    # all independently dialable -> attribute failures to specific nuisances
    lead_sil: int = 8           # mean leading silence (frames)
    lead_sil_jitter: int = 4    # +- uniform jitter on leading silence
    trail_sil: int = 4          # mean trailing silence
    dur_jitter: float = 0.0     # fractional per-phone duration jitter (=warp)
    p_drop: float = 0.0         # bit dropout prob (encoder miss)
    p_add: float = 0.0          # bit insertion prob (encoder false bit)
    p_sub: float = 0.0          # phone substitution prob (confusable)
    stability: float = 0.85     # prob a bit persists frame-to-frame within phone
    seed: int = 0


@dataclass
class VoteConfig:
    W: int = 12                 # gating window half-width (max plausible offset)
    splat: int = 1              # soft-bin half-width (sub-bin warp/jitter tol)
    lam: float = 0.85           # leak for the accumulator
    init_phase_unconstrained: bool = True   # acquisition: first frames vote wide


@dataclass
class TraversalConfig:
    """Hard monotone-traversal voting (position-vote + high-water-mark).

    Per query frame, each active feature votes GLOBALLY (no window) for stored
    positions where it occurs. The consensus landed position pi_t is the peak.
    Score rewards monotone FORWARD progress; backward/stationary gives nothing.
    Similarity = fraction of the stored sequence traversed in coherent order
    (high-water-mark / coverage), gated by per-frame match confidence.
    """
    min_conf: float = 2.0       # min peak PROMINENCE (peak / mean-bin) to accept
    max_jump: int = 0           # 0 = allow arbitrary forward jumps; >0 caps them
    back_tol: int = 1           # backward motion up to this is tolerated (noise)


@dataclass
class SoftFrontConfig:
    """Soft monotone-traversal: a distribution over PROGRESS, updated like an
    HMM forward pass. Keeps graded timing evidence the hard version discards.

    State: alpha[k, pos] = unnormalized belief that, after the frames seen so
    far, the query has progressed to stored position `pos` of sequence k, having
    traversed monotonically. Each frame:
      1. TRANSITION (monotone): belief can stay or advance, never go back.
         alpha <- forward-smear of alpha (geometric advance distribution).
      2. EMISSION: multiply by this frame's per-position vote evidence (how well
         the active features match each stored position).
      3. renormalize within the sequence; accumulate log-evidence as the score.
    Score = total forward log-evidence (how well a monotone path explains the
    query). Warp-tolerant (advance distribution absorbs rate), acquisition-free
    (initial alpha uniform over start positions), silence-robust (empty frames
    contribute flat evidence -> no score change).
    """
    advance_mean: float = 1.0   # expected positions advanced per query frame
    advance_spread: float = 2.0 # how far the forward transition can reach
    stay_prob: float = 0.4      # mass kept at current position (slow speech/holds)
    emission_floor: float = 0.05  # min per-position emission (noise robustness)
    leak_uniform: float = 0.02  # small uniform mixing (re-acquisition / recovery)


# ============================================================================
# 1. Generation: phones -> classes -> utterances  (TCN->SBDR encoder model)
# ============================================================================

def make_phone_bank(cfg: GenConfig, rng):
    """Each phone = a sparse binary prototype drawn from a SHARED feature pool,
    so phones (and thus classes) overlap partially in feature content."""
    pool = rng.choice(cfg.D, size=min(cfg.phone_pool, cfg.D), replace=False)
    phones = []
    for _ in range(cfg.n_phones):
        bits = rng.choice(pool, size=cfg.code_density, replace=False)
        v = np.zeros(cfg.D, dtype=bool)
        v[bits] = True
        phones.append(v)
    return np.stack(phones), pool


def make_classes(cfg: GenConfig, rng):
    """Each class = an ordered string of phone indices.

    Default: independently sampled strings (shared bank, but differing phone
    multisets -> partly content-separable).
    order_sensitive: every class is a distinct PERMUTATION of ONE shared phone
    multiset -> near-identical phone histograms, only ARRANGEMENT differs.
    """
    if cfg.order_sensitive:
        base = rng.integers(0, cfg.n_phones, size=cfg.phones_per_class)
        classes = []
        for _ in range(cfg.n_classes):
            classes.append(rng.permutation(base))
        return classes
    return [rng.integers(0, cfg.n_phones, size=cfg.phones_per_class)
            for _ in range(cfg.n_classes)]


def render_utterance(class_string, phones, gcfg: GenConfig, ucfg: UttConfig,
                     rng):
    """Render one noisy, time-varied utterance of a class into an SBDR sequence.

    Returns (seq[T,D] bool, meta) where meta carries the per-frame source phone
    and the ground-truth onset, for alignment/diagnostics.
    """
    D = gcfg.D
    frames = []
    src_phone = []   # ground-truth phone per frame (or -1 for silence)

    lead = max(0, ucfg.lead_sil + int(rng.integers(-ucfg.lead_sil_jitter,
                                                    ucfg.lead_sil_jitter + 1)))
    for _ in range(lead):
        frames.append(np.zeros(D, dtype=bool)); src_phone.append(-1)
    onset = len(frames)

    prev = np.zeros(D, dtype=bool)
    for pi in class_string:
        # phone substitution (confusable classes)
        if ucfg.p_sub > 0 and rng.random() < ucfg.p_sub:
            pi = rng.integers(0, gcfg.n_phones)
        proto = phones[pi]
        # duration jitter = local time warp (speaking rate)
        dur = max(1, int(round(gcfg.nominal_dur *
                               (1.0 + rng.normal(0, ucfg.dur_jitter)))))
        for f in range(dur):
            v = np.zeros(D, dtype=bool)
            active = np.flatnonzero(proto)
            # within-phone stability: persist some previously-on bits
            keep = np.flatnonzero(prev & proto)
            for b in active:
                on = True
                if f > 0 and b not in keep and rng.random() > ucfg.stability:
                    on = False  # not yet stabilized
                if ucfg.p_drop > 0 and rng.random() < ucfg.p_drop:
                    on = False  # encoder dropout
                if on:
                    v[b] = True
            # bit insertion (encoder false positives)
            if ucfg.p_add > 0:
                add = np.flatnonzero(rng.random(D) < ucfg.p_add)
                v[add] = True
            frames.append(v); src_phone.append(int(pi))
            prev = v

    for _ in range(ucfg.trail_sil):
        frames.append(np.zeros(D, dtype=bool)); src_phone.append(-1)

    seq = np.stack(frames) if frames else np.zeros((1, D), dtype=bool)
    return seq, {"onset": onset, "src_phone": np.array(src_phone)}


def to_active_sets(seq):
    return [np.flatnonzero(seq[t]) for t in range(seq.shape[0])]


def build_database(gcfg: GenConfig, store_ucfg: UttConfig):
    """One clean-ish stored template per class (the gallery)."""
    rng = np.random.default_rng(gcfg.seed)
    phones, pool = make_phone_bank(gcfg, rng)
    classes = make_classes(gcfg, rng)
    stored, metas = [], []
    for ci, cs in enumerate(classes):
        r = np.random.default_rng(gcfg.seed + 1000 + ci)
        seq, meta = render_utterance(cs, phones, gcfg, store_ucfg, r)
        stored.append(seq); metas.append(meta)
    return {"phones": phones, "pool": pool, "classes": classes,
            "stored": stored, "metas": metas, "gcfg": gcfg}


# ============================================================================
# 2. Stored-sequence preprocessing: inverted index
# ============================================================================

def build_index(seq):
    T, D = seq.shape
    inverted = [np.flatnonzero(seq[:, d]) for d in range(D)]
    return {"inverted": inverted, "T": T, "D": D,
            "active_sets": to_active_sets(seq)}


# ============================================================================
# 3. THE SINGLE CRITERION: leaky soft-binned offset-vote accumulator
# ============================================================================

class VoteAccumulator:
    """Per-sequence absolute-offset vote histogram, leaky and soft-binned.

    Offset bins span [-W, +W]. Each step:
      - predicted position p = current phase + 1 (nominal unit advance)
      - every active query feature votes (with a soft splat) at offset = t - p
        for each stored occurrence t of that feature within the window
      - accumulator <- lam * accumulator + votes
      - peak bin -> phase correction; peak prominence -> match quality
      - if no votes (silence): abstain, phase coasts by +1
    """

    def __init__(self, stored_indices, cfg: VoteConfig):
        self.idx = stored_indices
        self.cfg = cfg
        self.K = len(stored_indices)
        self.nbin = 2 * cfg.W + 1
        self.reset()

    def reset(self):
        self.acc = np.zeros((self.K, self.nbin))   # leaky offset histograms
        self.phase = np.zeros(self.K)
        self.quality = np.zeros(self.K)             # last-step match quality
        self.score = np.zeros(self.K)               # silence-robust readout
        # evidence accumulation over INFORMATIVE frames only (silence-invariant):
        # score = sum(quality * evidence_mass) / sum(evidence_mass), where
        # evidence_mass = number of matched features that frame. Silent frames
        # contribute nothing to either sum, so they neither help nor erode.
        self._evid_num = np.zeros(self.K)           # sum quality * mass
        self._evid_den = np.zeros(self.K)           # sum mass
        self._t = 0

    def _splat(self, hist, center_bin, weight):
        """Deposit a triangular splat of half-width `splat` around center_bin."""
        s = self.cfg.splat
        for o in range(-s, s + 1):
            b = center_bin + o
            if 0 <= b < self.nbin:
                hist[b] += weight * (1.0 - abs(o) / (s + 1))

    def step(self, q_active, record=False):
        cfg = self.cfg
        W = cfg.W
        self.acc *= cfg.lam                         # leak
        rec = {} if record else None

        for k in range(self.K):
            inv = self.idx[k]["inverted"]
            p = self.phase[k] + 1.0
            # collect, per active feature, its in-window candidate offsets.
            # KEY FIX: each feature contributes ONE unit of vote mass, split
            # across its candidate occurrences (recurrence -> diluted, not
            # amplified). A feature occurring once at one offset votes
            # decisively; a feature occurring everywhere is uninformative.
            frame_votes = np.zeros(self.nbin)
            feat_best_delta = []       # best (nearest) offset per matched feature
            n_matched = 0
            for d in q_active:
                if d >= len(inv):
                    continue
                times = inv[d]
                if times.size == 0:
                    continue
                lo = np.searchsorted(times, p - W, side="left")
                hi = np.searchsorted(times, p + W, side="right")
                cand = times[lo:hi]
                if cand.size == 0:
                    continue
                w = 1.0 / cand.size            # normalized vote mass
                nearest = cand[np.argmin(np.abs(cand - p))]
                feat_best_delta.append(int(round(nearest - p)))
                n_matched += 1
                for t in cand:
                    delta = int(round(t - p))
                    if -W <= delta <= W:
                        self._splat(frame_votes, delta + W, w)

            self.acc[k] += frame_votes
            hist = self.acc[k]
            total = hist.sum()

            if n_matched > 0 and total > 1e-9:
                peak_bin = int(np.argmax(hist))
                delta_star = peak_bin - W
                # BOUNDED quality: fraction of THIS frame's matched features
                # whose nearest offset agrees with the consensus offset
                # (within the splat tolerance). Independent of accumulator mass.
                tol = cfg.splat
                agree = sum(1 for db_ in feat_best_delta
                            if abs(db_ - delta_star) <= tol)
                recall = n_matched / max(len(q_active), 1)
                consensus = agree / max(n_matched, 1)
                quality = recall * consensus
                self.phase[k] = p + delta_star
            else:
                quality = 0.0
                self.phase[k] = p

            self.quality[k] = quality
            # silence-robust readout: accumulate evidence over informative
            # frames only, weighted by evidence mass (n_matched). Silent frames
            # (n_matched==0) skip both sums -> invisible to the score.
            if n_matched > 0:
                self._evid_num[k] += quality * n_matched
                self._evid_den[k] += n_matched
            self.score[k] = (self._evid_num[k] / self._evid_den[k]
                             if self._evid_den[k] > 0 else 0.0)

            if record:
                rec.setdefault("peak", np.zeros(self.K))
                rec.setdefault("total", np.zeros(self.K))
                rec.setdefault("delta_star", np.zeros(self.K))
                rec["peak"][k] = hist.max() if total > 0 else 0.0
                rec["total"][k] = total
                rec["delta_star"][k] = (int(np.argmax(hist)) - W) if total > 0 else 0

        self._t += 1
        out = {"phase": self.phase.copy(), "quality": self.quality.copy(),
               "score": self.score.copy()}
        if record:
            out.update(rec)
        return out

    def predict_next(self, k):
        pos = int(round(self.phase[k] + 1.0))
        sets = self.idx[k]["active_sets"]
        if 0 <= pos < len(sets):
            return sets[pos]
        return np.array([], dtype=int)

    def scores(self):
        return self.score.copy()


# ============================================================================
# 3b. ALTERNATIVE SINGLE CRITERION: hard monotone-traversal voting
# ============================================================================

class TraversalAccumulator:
    """Position-voting + high-water-mark monotone traversal.

    Per query frame, each active feature votes GLOBALLY (no window) for every
    stored position where it occurs, with per-feature normalized mass (recurring
    features diluted, as in the offset version). The consensus landed position
    pi_t = argmax of the position histogram, with a confidence = peak/total.

    Scoring (the single criterion): track a high-water mark hwm_k = furthest
    stored position reached in coherent forward order. Each confident frame:
      - forward move (pi_t > hwm - back_tol): advance hwm to pi_t; credit the
        amount of NEW ground covered.
      - backward/stationary: no credit (revisiting loses similarity).
      - optional max_jump cap: forward jumps beyond it are not credited (skipping
        too much content shouldn't count as coverage).
    Similarity = covered_ground / stored_length  in [0, 1]  (fraction of the
    sequence traversed monotonically). Free acquisition (no window), warp-free
    (only order matters, not rate), arbitrary forward jumps allowed.
    """

    def __init__(self, stored_indices, cfg: TraversalConfig):
        self.idx = stored_indices
        self.cfg = cfg
        self.K = len(stored_indices)
        self.T = [ix["T"] for ix in stored_indices]
        self.reset()

    def reset(self):
        self.hwm = np.full(self.K, -1.0)     # high-water mark per sequence
        self.covered = np.zeros(self.K)      # monotone ground covered
        self.pi = np.full(self.K, -1.0)      # last landed position
        self.conf = np.zeros(self.K)         # last confidence
        self.score = np.zeros(self.K)
        self._t = 0

    def step(self, q_active, record=False):
        cfg = self.cfg
        rec = {} if record else None
        for k in range(self.K):
            inv = self.idx[k]["inverted"]
            T = self.T[k]
            hist = np.zeros(T)
            n_matched = 0
            for d in q_active:
                if d >= len(inv):
                    continue
                times = inv[d]
                if times.size == 0:
                    continue
                w = 1.0 / times.size           # normalized vote mass
                hist[times] += w
                n_matched += 1

            total = hist.sum()
            if n_matched > 0 and total > 1e-9:
                pi_t = int(np.argmax(hist))
                peak = hist[pi_t]
                # confidence as PROMINENCE over the histogram's own baseline:
                # peak relative to the mean nonzero bin mass. Scale-appropriate
                # for the full-length position histogram (unlike peak/total,
                # which is tiny simply because there are many bins). conf>1 means
                # the peak stands above a flat distribution; higher = sharper.
                nonzero = np.count_nonzero(hist)
                baseline = total / max(nonzero, 1)
                conf = peak / baseline if baseline > 0 else 0.0
            else:
                pi_t, conf = -1, 0.0

            self.pi[k] = pi_t
            self.conf[k] = conf

            # traversal scoring: credit monotone forward coverage only
            if pi_t >= 0 and conf >= cfg.min_conf:
                if self.hwm[k] < 0:
                    # acquisition: land anywhere, start the mark here
                    self.hwm[k] = pi_t
                else:
                    advance = pi_t - self.hwm[k]
                    forward_ok = advance > 0
                    jump_ok = (cfg.max_jump == 0) or (advance <= cfg.max_jump)
                    back_ok = advance >= -cfg.back_tol
                    if forward_ok and jump_ok:
                        # credit new ground covered (raw forward distance)
                        self.covered[k] += advance
                        self.hwm[k] = pi_t
                    elif back_ok and advance <= 0:
                        pass  # tolerated jitter / stationary: no credit, no loss
                    else:
                        # backward jump (revisit) or oversized skip: no credit
                        pass

            self.score[k] = self.covered[k] / max(T, 1)

            if record:
                for key in ("pi", "conf", "hwm", "covered", "score"):
                    rec.setdefault(key, np.zeros(self.K))
                rec["pi"][k] = pi_t
                rec["conf"][k] = conf
                rec["hwm"][k] = self.hwm[k]
                rec["covered"][k] = self.covered[k]
                rec["score"][k] = self.score[k]

        self._t += 1
        out = {"pi": self.pi.copy(), "conf": self.conf.copy(),
               "score": self.score.copy()}
        if record:
            out.update(rec)
        return out

    def predict_next(self, k):
        pos = int(round(self.pi[k] + 1.0)) if self.pi[k] >= 0 else 0
        sets = self.idx[k]["active_sets"]
        if 0 <= pos < len(sets):
            return sets[pos]
        return np.array([], dtype=int)

    def scores(self):
        return self.score.copy()


# ============================================================================
# 3c. SOFT progress-front: HMM-forward over monotone progress
# ============================================================================

class SoftFrontAccumulator:
    """Soft monotone-traversal via an HMM-style forward pass over progress.

    Per sequence k, maintain alpha[pos] over stored positions = belief of having
    monotonically progressed to `pos`. Each query frame:
      1. emission e[pos] = per-position vote evidence (normalized, floored)
      2. transition: alpha <- monotone forward smear of alpha (stay or advance)
      3. alpha <- alpha * e ; accumulate log(sum) as forward evidence; renormalize
    Score = mean per-informative-frame forward log-evidence (silence-robust).
    Retains graded timing (unlike the hard high-water-mark) while staying
    warp- and acquisition-tolerant.
    """

    def __init__(self, stored_indices, cfg: SoftFrontConfig):
        self.idx = stored_indices
        self.cfg = cfg
        self.K = len(stored_indices)
        self.T = [ix["T"] for ix in stored_indices]
        # precompute the monotone forward transition kernel (geometric-ish):
        # mass to advance by j positions, j=0..spread. j=0 is "stay".
        s = cfg.advance_spread
        kern = np.array([cfg.stay_prob] +
                        [(1 - cfg.stay_prob) * np.exp(-abs(j - cfg.advance_mean) / s)
                         for j in range(1, int(s) + 2)])
        self.kernel = kern / kern.sum()
        self.reset()

    def reset(self):
        # alpha starts uniform over a small prefix (acquisition-free: query may
        # begin anywhere near the start; silence frames keep it diffuse).
        self.alpha = [np.ones(T) / T for T in self.T]
        self.logevid = np.zeros(self.K)     # accumulated forward log-evidence
        self.n_inform = np.zeros(self.K)    # informative-frame count
        self.score = np.zeros(self.K)
        self.front = np.zeros(self.K)       # expected position (for diagnostics)
        self._t = 0

    def _emission(self, k, q_active):
        """Per-position evidence: normalized vote mass each stored position gets
        from the active query features (recurrence-diluted), floored for noise."""
        inv = self.idx[k]["inverted"]
        T = self.T[k]
        e = np.zeros(T)
        n_matched = 0
        for d in q_active:
            if d >= len(inv):
                continue
            times = inv[d]
            if times.size == 0:
                continue
            e[times] += 1.0 / times.size
            n_matched += 1
        return e, n_matched

    def _transition(self, alpha):
        """Monotone forward smear: convolve alpha with the forward kernel so
        belief can only stay or advance (never move backward)."""
        out = np.zeros_like(alpha)
        for j, w in enumerate(self.kernel):
            if w == 0:
                continue
            if j == 0:
                out += w * alpha
            else:
                out[j:] += w * alpha[:-j]
                # mass that would advance past the end piles up at the end
                out[-1] += w * alpha[-j:].sum() if j <= len(alpha) else 0.0
        return out

    def step(self, q_active, record=False):
        cfg = self.cfg
        rec = {} if record else None
        for k in range(self.K):
            T = self.T[k]
            e, n_matched = self._emission(k, q_active)

            if n_matched > 0 and e.sum() > 1e-9:
                # normalize emission to a distribution, floor for robustness
                e = e / e.sum()
                e = e + cfg.emission_floor / T
                e = e / e.sum()
                informative = True
            else:
                # silence / no match: flat emission -> alpha only transitions
                e = np.ones(T) / T
                informative = False

            a = self.alpha[k]
            a = self._transition(a)                  # monotone forward
            # small uniform mixing aids recovery from a bad lock
            a = (1 - cfg.leak_uniform) * a + cfg.leak_uniform / T
            unnorm = a * e
            mass = unnorm.sum()
            if mass > 1e-12:
                if informative:
                    self.logevid[k] += np.log(mass * T)  # vs uniform baseline
                    self.n_inform[k] += 1
                self.alpha[k] = unnorm / mass
            else:
                self.alpha[k] = np.ones(T) / T

            self.front[k] = float(np.dot(self.alpha[k], np.arange(T)))
            self.score[k] = (self.logevid[k] / self.n_inform[k]
                             if self.n_inform[k] > 0 else 0.0)

            if record:
                for key in ("front", "score", "logevid"):
                    rec.setdefault(key, np.zeros(self.K))
                rec["front"][k] = self.front[k]
                rec["score"][k] = self.score[k]
                rec["logevid"][k] = self.logevid[k]

        self._t += 1
        out = {"front": self.front.copy(), "score": self.score.copy()}
        if record:
            out.update(rec)
        return out

    def predict_next(self, k):
        pos = int(round(self.front[k] + cfg_advance(self)))
        sets = self.idx[k]["active_sets"]
        if 0 <= pos < len(sets):
            return sets[pos]
        return np.array([], dtype=int)

    def scores(self):
        return self.score.copy()


def cfg_advance(acc):
    return acc.cfg.advance_mean


def bag_of_features_score(stored_seqs, query):
    q = query.sum(axis=0).astype(float); qn = np.linalg.norm(q) + 1e-9
    best, bk = -1, -1
    for k, s in enumerate(stored_seqs):
        v = s.sum(axis=0).astype(float)
        c = float(q @ v) / ((np.linalg.norm(v) + 1e-9) * qn)
        if c > best: best, bk = c, k
    return bk


def single_offset_xcorr_score(stored_seqs, query, max_off=15):
    """Best single global offset by set-overlap (ablation: no temporal accum)."""
    q_sets = to_active_sets(query)
    qsets = [set(s.tolist()) for s in q_sets]
    best, bk = -1, -1
    for k, s in enumerate(stored_seqs):
        s_sets = [set(np.flatnonzero(s[t]).tolist()) for t in range(s.shape[0])]
        best_off = 0
        for off in range(-max_off, max_off + 1):
            tot = 0; n = 0
            for ti in range(len(qsets)):
                tj = ti + off
                if 0 <= tj < len(s_sets):
                    a, b = qsets[ti], s_sets[tj]
                    if a or b:
                        tot += len(a & b) / len(a | b); n += 1
            score = tot / max(n, 1)
            if score > best_off: best_off = score
        if best_off > best: best, bk = best_off, k
    return bk


def windowed_dtw_score(stored_seqs, query, band=15):
    """Sakoe-Chiba banded DTW on (1 - Jaccard) frame distance. Warp-optimal
    ceiling; O(T*band) per pair. Returns argmin-distance class."""
    q_sets = [set(np.flatnonzero(query[t]).tolist()) for t in range(query.shape[0])]
    best, bk = np.inf, -1
    for k, s in enumerate(stored_seqs):
        s_sets = [set(np.flatnonzero(s[t]).tolist()) for t in range(s.shape[0])]
        n, m = len(q_sets), len(s_sets)
        INF = np.inf
        prev = np.full(m + 1, INF); prev[0] = 0
        for i in range(1, n + 1):
            cur = np.full(m + 1, INF)
            jlo = max(1, i - band); jhi = min(m, i + band)
            for j in range(jlo, jhi + 1):
                a, b = q_sets[i - 1], s_sets[j - 1]
                d = 1.0 - (len(a & b) / len(a | b)) if (a or b) else 0.0
                cur[j] = d + min(prev[j], cur[j - 1], prev[j - 1])
            prev = cur
        dist = prev[m] / (n + m)
        if dist < best: best, bk = dist, k
    return bk


# ============================================================================
# 5. Harness + evaluation
# ============================================================================

def f1_sets(pred, true):
    if len(pred) == 0 and len(true) == 0: return 1.0
    if len(pred) == 0 or len(true) == 0: return 0.0
    inter = len(np.intersect1d(pred, true))
    prec = inter / len(pred); rec = inter / len(true)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def run_query(acc, query, true_k, record=False):
    """Stream a query through any accumulator (offset or traversal).
    Record keys are derived from the accumulator's own step() output."""
    acc.reset()
    q_sets = to_active_sets(query)
    T = len(q_sets)
    pred_f1 = np.full(T, np.nan)
    hist = None
    for t in range(T):
        pred = acc.predict_next(true_k)
        if t + 1 < T:
            pred_f1[t] = f1_sets(pred, q_sets[t + 1])
        out = acc.step(q_sets[t], record=record)
        if record:
            if hist is None:
                hist = {key: np.zeros((T, acc.K)) for key in out
                        if isinstance(out[key], np.ndarray)}
            for key in hist:
                hist[key][t] = out[key]
    sc = acc.scores()
    pred_k = int(np.argmax(sc))
    order = np.argsort(sc)[::-1]
    margin = sc[order[0]] - sc[order[1]] if len(sc) > 1 else 0.0
    res = {"pred_k": pred_k, "correct": pred_k == true_k, "margin": float(margin),
           "pred_f1": pred_f1, "scores": sc}
    if record:
        hist["pred_f1"] = pred_f1; hist["q_sets"] = q_sets
        res["hist"] = hist
    return res


def make_query(db, true_k, ucfg: UttConfig, seed):
    r = np.random.default_rng(seed)
    return render_utterance(db["classes"][true_k], db["phones"],
                            db["gcfg"], ucfg, r)


def sweep(db, acc_factory, base_ucfg, param, levels, n_trials=10,
          baselines=("bof", "xcorr"), base_seed=0):
    """Sweep one UttConfig parameter; compare a voting accumulator vs baselines.

    acc_factory(indices) -> accumulator instance (lets us swap offset vs
    traversal criteria without changing the harness).
    """
    indices = [build_index(s) for s in db["stored"]]
    acc = acc_factory(indices)
    K = len(db["stored"])
    rows = []
    for lev in levels:
        for trial in range(n_trials):
            true_k = trial % K
            ucfg = UttConfig(**{**base_ucfg.__dict__, param: lev,
                                "seed": base_seed * 1000 + trial})
            query, meta = make_query(db, true_k, ucfg, ucfg.seed)
            res = run_query(acc, query, true_k)
            row = {param: lev, "trial": trial, "correct": res["correct"],
                   "margin": res["margin"],
                   "pred_f1": np.nanmean(res["pred_f1"])}
            if "bof" in baselines:
                row["bof"] = bag_of_features_score(db["stored"], query) == true_k
            if "xcorr" in baselines:
                row["xcorr"] = single_offset_xcorr_score(db["stored"], query) == true_k
            if "dtw" in baselines:
                row["dtw"] = windowed_dtw_score(db["stored"], query) == true_k
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# 6. Visualization
# ============================================================================

def plot_accumulator_evolution(hist, true_k, vote_cfg, path="vv_accum.png"):
    """Heatmap of the true sequence's offset accumulator over time + the read-off
    quantities (phase, quality, peak prominence, predictive F1)."""
    import matplotlib.pyplot as plt
    T = hist["phase"].shape[0]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(hist["delta_star"][:, true_k], color="tab:blue")
    axes[0].axhline(0, color="k", ls=":", alpha=0.5)
    axes[0].set_ylabel("consensus\noffset δ*")
    axes[0].set_title(f"Vote accumulator behavior (true class = {true_k})")

    axes[1].plot(hist["quality"][:, true_k], color="tab:orange", label="per-step quality")
    axes[1].plot(hist["score"][:, true_k], color="tab:red", label="leaky score")
    axes[1].set_ylabel("match quality"); axes[1].legend(fontsize=7)

    # peak prominence = peak / total
    prom = hist["peak"][:, true_k] / np.maximum(hist["total"][:, true_k], 1e-9)
    axes[2].plot(prom, color="tab:green")
    axes[2].set_ylabel("peak\nprominence")
    axes[2].set_ylim(0, 1.05)

    axes[3].plot(hist["pred_f1"], color="tab:purple")
    axes[3].set_ylabel("predictive F1"); axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel("time (query frames)")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig); return path


def plot_score_landscape(hist, true_k, top_n=6, path="vv_scores.png"):
    """Leaky score trajectories of the top sequences; true one bold."""
    import matplotlib.pyplot as plt
    sc = hist["score"]
    top = list(np.argsort(sc[-1])[::-1][:top_n])
    if true_k not in top: top = [true_k] + top[:top_n - 1]
    fig, ax = plt.subplots(figsize=(11, 4))
    for k in top:
        lw = 2.5 if k == true_k else 1.0
        ax.plot(sc[:, k], lw=lw,
                label=f"class {k}" + (" (true)" if k == true_k else ""))
    ax.set_xlabel("time (query frames)"); ax.set_ylabel("leaky score")
    ax.legend(fontsize=7, ncol=top_n); ax.set_title("Score landscape over time")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig); return path


def plot_phones_and_utterance(db, true_k, query, path="vv_data.png"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].imshow(db["stored"][true_k].T, aspect="auto", cmap="Greys",
                   interpolation="nearest")
    axes[0].set_title(f"Stored template, class {true_k} (feature x time)")
    axes[0].set_ylabel("feature")
    axes[1].imshow(query.T, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[1].set_title("Noisy query utterance of the same class")
    axes[1].set_ylabel("feature"); axes[1].set_xlabel("time (frames)")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig); return path


def plot_soft_front(db, true_k, query, soft_cfg, path="vv_softfront.png"):
    """Show the soft progress-front: alpha distribution over stored positions
    evolving with query time (heatmap), the expected front, and the score gap
    between the true class and its best competitor."""
    import matplotlib.pyplot as plt
    indices = [build_index(s) for s in db["stored"]]
    acc = SoftFrontAccumulator(indices, soft_cfg)
    # record the alpha field of the true class over time
    acc.reset()
    q_sets = to_active_sets(query)
    T = len(q_sets); Tk = acc.T[true_k]
    alpha_field = np.zeros((T, Tk))
    fronts = np.zeros(T)
    score_true = np.zeros(T); score_best_other = np.zeros(T)
    for t in range(T):
        acc.step(q_sets[t])
        alpha_field[t] = acc.alpha[true_k]
        fronts[t] = acc.front[true_k]
        sc = acc.score.copy()
        score_true[t] = sc[true_k]
        other = np.delete(sc, true_k)
        score_best_other[t] = other.max()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].imshow(alpha_field.T, aspect="auto", origin="lower",
                   cmap="viridis", interpolation="nearest")
    axes[0].plot(np.arange(T), fronts, color="white", lw=1.5, alpha=0.8,
                 label="expected front")
    axes[0].set_ylabel("stored position")
    axes[0].set_title(f"Soft progress-front: belief over position, class {true_k}")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(score_true, color="tab:red", lw=2, label="true class score")
    axes[1].plot(score_best_other, color="tab:gray", lw=1,
                 label="best competitor")
    axes[1].set_xlabel("query frame"); axes[1].set_ylabel("forward log-evid")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig); return path


def run_visualizations():
    gcfg = GenConfig(order_sensitive=True)
    db = build_database(gcfg, UttConfig(p_drop=0.05))
    vote_cfg = VoteConfig(); soft_cfg = SoftFrontConfig()
    indices = [build_index(s) for s in db["stored"]]
    true_k = 3
    ucfg = UttConfig(lead_sil=12, dur_jitter=0.2, p_drop=0.1, p_add=0.01, seed=7)
    query, meta = make_query(db, true_k, ucfg, ucfg.seed)

    # offset-voting diagnostics (existing)
    acc = VoteAccumulator(indices, vote_cfg)
    res = run_query(acc, query, true_k, record=True)
    paths = [
        plot_phones_and_utterance(db, true_k, query),
        plot_accumulator_evolution(res["hist"], true_k, vote_cfg),
        plot_soft_front(db, true_k, query, soft_cfg),
    ]
    print("Saved:", *paths, sep="\n  ")
    return paths


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    import sys
    if "--viz" in sys.argv:
        run_visualizations(); sys.exit(0)

    gcfg_os = GenConfig(order_sensitive=True)
    db_os = build_database(gcfg_os, UttConfig(p_drop=0.05))
    vote_cfg = VoteConfig()
    trav_cfg = TraversalConfig()
    soft_cfg = SoftFrontConfig()
    base = UttConfig()

    offset_fac = lambda idx: VoteAccumulator(idx, vote_cfg)
    trav_fac = lambda idx: TraversalAccumulator(idx, trav_cfg)
    soft_fac = lambda idx: SoftFrontAccumulator(idx, soft_cfg)

    def show(df, p, label):
        cols = {"acc": ("correct", "mean"), "margin": ("margin", "mean")}
        for b in ("bof", "xcorr", "dtw"):
            if b in df.columns: cols[b] = (b, "mean")
        print(f"  [{label}]")
        print(df.groupby(p).agg(**{k: v for k, v in cols.items()}
                                ).round(3).to_string())

    def compare(param, levels, baselines, title):
        print(f"\n=== {title} (ORDER-sensitive) ===")
        show(sweep(db_os, offset_fac, base, param, levels, n_trials=20,
                   baselines=baselines), param, "OFFSET voting")
        show(sweep(db_os, trav_fac, base, param, levels, n_trials=20,
                   baselines=()), param, "TRAVERSAL (hard)")
        show(sweep(db_os, soft_fac, base, param, levels, n_trials=20,
                   baselines=()), param, "SOFT front")

    compare("dur_jitter", [0.0, 0.1, 0.2, 0.3, 0.4],
            ("bof", "xcorr", "dtw"), "Duration jitter (warp)")
    compare("p_drop", [0.0, 0.1, 0.2, 0.3, 0.4],
            ("bof", "xcorr"), "Bit dropout (encoder miss)")
    compare("lead_sil", [0, 8, 16, 24, 32],
            ("bof", "xcorr"), "Leading silence (acquisition)")

    print("\ncols: acc=method top-1, bof/xcorr/dtw=baseline top-1, "
          "margin=score gap")

    print("\ncols: acc=voting top-1, bof/xcorr/dtw=baseline top-1, "
          "margin=score gap, pred_f1=predictive F1")