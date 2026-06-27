# The Soft Progress-Front: An Online Criterion for Sparse Binary Sequence Retrieval

## Purpose

This document specifies a single, self-contained matching criterion for comparing a streaming **query sequence** of sparse binary vectors against a database of stored **template sequences**, online and at low cost. It produces, at every timestep and for every stored template, a similarity score that is:

- **warp-tolerant** — invariant to changes in speaking rate / playback speed,
  whether uniform or mildly nonlinear;
- **acquisition-free** — robust to arbitrary, differing amounts of leading   silence or onset offset, with no alignment window to set;
- **silence-robust** — silent or uninformative frames neither help nor harm the   score;
- **noise-tolerant** — robust to dropped and spuriously inserted active bits, as   produced by a real encoder;
- **discriminative through order** — it separates templates that share the same   feature content but differ in temporal *arrangement*, which content-only   measures (bag-of-features) cannot.

It requires no per-dataset weight tuning: a single criterion, with a handful of interpretable parameters that proved insensitive in testing.

The data is assumed to be **Sparse Binary Distributed Representations (SBDR)**:
each frame is a binary vector in $\{0,1\}^D$ with only $a \ll D$ active bits.
Such codes arise, for example, from a separately trained encoder (e.g. a TCN)
mapping each frame — or pool of frames — of some signal into a sparse binary code. The criterion is agnostic to the signal domain (audio, vision, robot perception); it operates purely on the SBDR sequences.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $D$ | dimensionality of the SBDR space |
| $a$ | number of active bits per frame (sparsity), $a \ll D$ |
| $K$ | number of stored template sequences |
| $T_k$ | length (frames) of stored template $k$ |
| $s^{(k)}_p \in \{0,1\}^D$ | frame $p$ of stored template $k$ |
| $A^{(k)}_p = \{d : s^{(k)}_{p,d} = 1\}$ | active set of stored frame $p$ |
| $q_t \in \{0,1\}^D$ | frame $t$ of the streaming query |
| $Q_t = \{d : q_{t,d} = 1\}$ | active set of query frame $t$ |
| $P^{(k)}_d = \{p : d \in A^{(k)}_p\}$ | **occurrence list**: stored positions where feature $d$ is active |

The occurrence lists $P^{(k)}_d$ form an **inverted index** of the database. They are the single structure that makes the criterion cheap, and they are computed once, offline.

---

## 2. The core idea

We model the query as a particle that **traverses** a stored template from start to end. If the query truly matches template $k$, then as query time $t$
advances, the position $p$ in the template that best explains the current query frame should advance too — **monotonically**, never moving backward, regardless of *how fast* it advances. A wrong template may explain individual frames (shared features), but the explained positions will not form a coherent forward path.

So the discriminator is not "do the features match?" but "**do the matched positions sweep through the template in order?**" This reframing is what gives warp tolerance for free: monotone forward motion is invariant to rate.

Rather than commit to a single best position each frame (a brittle hard decision), we maintain a **distribution over progress** — a belief about how far through the template the query has advanced — and update it like the forward pass of a Hidden Markov Model. The progress distribution is the *soft progress-front*.

---

## 3. The state: a progress distribution 
For each stored template $k$, the state at query time $t$ is a vector 
$$
\boldsymbol{\alpha}^{(k)}_t \in \mathbb{R}^{T_k}_{\ge 0}, \qquad
\sum_{p=0}^{T_k - 1} \alpha^{(k)}_t[p] = 1 .
$$

$\alpha^{(k)}_t[p]$ is the belief that, having consumed query frames
$0, \dots, t$, the query has progressed **monotonically** to stored position $p$.

**Initialization (acquisition-free).** Before any frame is seen, we place a uniform prior,

$$
\alpha^{(k)}_{-1}[p] = \frac{1}{T_k} \quad \text{for all } p .
$$

A uniform prior means the query is allowed to begin anywhere; there is no preferred onset and no window. Variable leading silence is therefore not a special case — during silent frames the belief simply remains diffuse (see §6),
and it concentrates only once real evidence arrives.

---

## 4. The two update operators

Each query frame triggers a **transition** (enforcing monotone progress) followed by an **emission** (injecting the current frame's evidence). This is exactly the HMM forward recursion, specialized so the hidden state is *progress* and the transition is *forward-only*.

### 4.1 Transition: monotone forward smear

Progress may **stay** or **advance**, never retreat. We encode this with a fixed forward kernel $\boldsymbol{w} = (w_0, w_1, \dots, w_J)$, where $w_j$ is the probability of advancing by $j$ positions in one query frame:

$$
\tilde\alpha[p] \;=\; \sum_{j=0}^{J} w_j \, \alpha[p - j]
\qquad (\text{terms with } p - j < 0 \text{ omitted}).
$$

Because the sum runs only over $j \ge 0$, belief can only move **forward or stay** — monotonicity is built into the operator, not enforced by a penalty.

The kernel used here is a "stay-or-geometric-advance" shape:

$$
w_0 = \rho \ \ (\text{stay probability}), \qquad
w_j \propto (1-\rho)\, e^{-|j - \mu| / \sigma}\ \ (j \ge 1),
$$

then normalized so $\sum_j w_j = 1$. Here $\mu$ is the expected advance per query frame (nominally $1$), $\sigma$ controls how far a single frame may jump, and $\rho$ is the mass that stays in place (modeling slow speech / sustained states). Mass that would advance past the final position $T_k - 1$ accumulates there (an absorbing end).

**Why this gives warp tolerance.** A faster query advances more positions per frame; a slower query advances fewer. The kernel's spread $\sigma$ lets the front advance by a variable number of positions each frame, so a range of rates all produce valid monotone progress. Rate is absorbed by the transition, not penalized. No velocity needs to be estimated and no slope parameter is searched.

A small uniform mixing is applied after the transition, to allow recovery from a confidently-wrong lock:

$$
\tilde\alpha \;\leftarrow\; (1 - \eta)\,\tilde\alpha \;+\; \frac{\eta}{T_k}\,\mathbf{1},
$$

with $\eta$ small (e.g. $0.02$).

### 4.2 Emission: per-position vote evidence

The emission vector $\boldsymbol{e}_t \in \mathbb{R}^{T_k}_{\ge 0}$ scores how well the current query frame $Q_t$ matches each stored position $p$. This is computed by **voting through the inverted index**, and it is where sparsity makes the method cheap.

Each active query feature $d \in Q_t$ casts votes for the stored positions where it occurs, $P^{(k)}_d$. Crucially, each feature contributes **one unit of total vote mass**, split evenly across its occurrences:

$$
e_t[p] \;=\; \sum_{d \in Q_t}\ \frac{1}{\,|P^{(k)}_d|\,}\ \sum_{p' \in P^{(k)}_d} \mathbb{1}[p' = p]
\;=\; \sum_{\substack{d \in Q_t \\ p \in P^{(k)}_d}} \frac{1}{|P^{(k)}_d|}.
$$

The normalization $1/|P^{(k)}_d|$ is essential. Without it, a feature that recurs many times in a template would dominate the histogram regardless of alignment (the classic failure of unnormalized Hough voting). With it, a feature occurring **once** at one position votes decisively for that position, while a feature occurring **everywhere** spreads its single unit thinly and is correctly treated as uninformative. This is precisely an inverse-frequency (TF-IDF-like) weighting,
and it arises naturally from the "one feature, one vote" principle.

The raw emission is then turned into a likelihood-like distribution over positions: normalized to sum to one, mixed with a small floor $\epsilon$ for robustness to missing/aberrant evidence, and renormalized:

$$
\boldsymbol{e}_t \;\leftarrow\; \frac{\boldsymbol{e}_t}{\sum_p e_t[p]}, \qquad
\boldsymbol{e}_t \;\leftarrow\; \frac{\boldsymbol{e}_t + \epsilon / T_k}{\,1 + \epsilon\,}.
$$

The floor $\epsilon$ guarantees every position retains nonzero likelihood, so a single noisy frame can never zero out the belief — important under encoder dropout.

### 4.3 Combine: the forward recursion

The new belief multiplies transition by emission, then renormalizes:

$$
\hat\alpha_t[p] \;=\; \tilde\alpha_t[p] \cdot e_t[p], \qquad
m_t \;=\; \sum_p \hat\alpha_t[p], \qquad
\alpha^{(k)}_t \;=\; \frac{\hat\alpha_t}{m_t}.
$$

The scalar $m_t$ — the total mass before renormalization — is the **evidence**
this frame contributes: how well the frame's emission agreed with where the forward-transported belief expected the query to be. A coherent match produces a large $m_t$ (emission peaks where belief already concentrated); an incoherent one produces a small $m_t$ (emission lands where belief was thin). This single scalar is the heart of the score.

---

## 5. The score

We accumulate the log-evidence across query frames, measured **relative to a uniform baseline** (so that a frame which discriminates nothing contributes zero):

$$
\ell_t \;=\; \log\!\big(m_t \cdot T_k\big).
$$

The factor $T_k$ rescales so that $m_t = 1/T_k$ (uniform, uninformative) gives $\ell_t = 0$; concentrated agreement gives $\ell_t > 0$.

The final similarity of the query to template $k$ is the **mean log-evidence over informative frames only**:

$$
\boxed{\;\;
S^{(k)} \;=\; \frac{1}{|\mathcal{I}|} \sum_{t \in \mathcal{I}} \ell_t
\;\;}
$$

where $\mathcal{I}$ is the set of *informative* query frames — those with at least one active feature that matched the template (§6). Averaging over informative frames, rather than all frames, is what makes the score **silence-robust**: empty or unmatched frames are excluded from both numerator and denominator, so they cannot dilute the score.

The predicted class is $\hat k = \arg\max_k S^{(k)}$. The gap between the best and second-best $S^{(k)}$ is a natural, continuous **margin** / confidence.

### Why this discriminates by order

Two templates sharing the same feature *content* will both produce reasonable emissions frame by frame. But only the template whose positions are encountered in the correct *order* will have emissions that repeatedly land where the forward-only transition has already carried the belief. For that template the $m_t$ are consistently large and the log-evidence compounds. For a content-matched but order-wrong template, emissions scatter across positions the monotone transition cannot reach, the $m_t$ stay small, and the evidence plateaus. The score difference is produced by **path coherence**, not by content — which is exactly the discrimination that bag-of-features and single-offset methods lack.

---

## 6. Handling silence and uninformative frames

A query frame is **informative** for template $k$ if at least one of its active features occurs anywhere in that template (i.e. $\exists\, d \in Q_t$ with $P^{(k)}_d \neq \emptyset$). Otherwise — a silent frame, or a frame whose features are all absent from the template — the emission is set to uniform:

$$
\boldsymbol{e}_t = \tfrac{1}{T_k}\mathbf{1}.
$$

A uniform emission leaves the belief shape unchanged except for the transition (the front continues to drift forward at the kernel's expected rate), and it contributes **no evidence** ($\ell_t$ is not added; $t \notin \mathcal{I}$). Thus:

- **Leading silence** keeps the prior diffuse and the score untouched; the front   begins to concentrate only when content arrives, wherever that is.
- **Trailing silence** cannot erode an already-accumulated score.
- **Internal gaps** are bridged: the front coasts forward through the gap and   re-locks when content resumes.

This is the structural reason the criterion needs no onset detection and no alignment window.

---

## 7. Where computational efficiency comes from

The efficiency rests entirely on **sparsity** plus the **inverted index**.

### 7.1 The emission is the only nontrivial cost, and it is sparse

Computing $\boldsymbol{e}_t$ for one template touches only the occurrence lists of the $a = |Q_t|$ active query features. For each active feature $d$ we add $1/|P^{(k)}_d|$ to the positions in $P^{(k)}_d$ — a scatter-add into the emission vector. We never iterate over the $D$-dimensional space, and we never touch positions that no active feature votes for. The cost is 
$$
O\!\Big( \sum_{d \in Q_t} |P^{(k)}_d| \Big),
$$

i.e. proportional to the **total number of (feature, occurrence) incidences**
contributed by the currently-active features — which is small because $a$ is small and each feature's occurrence list is short in a sparse template. A dense frame-by-frame comparison, by contrast, would cost $O(T_k \cdot D)$ or $O(T_k \cdot a)$ per frame; the inverted index removes the factor of $T_k$
because each feature jumps directly to exactly the positions it affects.

### 7.2 The transition is a tiny fixed convolution

The forward smear convolves $\boldsymbol{\alpha}$ with a kernel of length $J + 1$, where $J$ is a small constant (a few positions). Its cost is $O(J \cdot T_k)$, and $J$ does not grow with anything. In practice the transition is cheaper than the emission and can be implemented as a handful of shifted adds.

### 7.3 Binary values make voting exact and branch-free

Because frames are **binary**, "match" is set membership: a feature is either active or not, and its contribution is either cast or not. There are no real-valued dot products, no per-element multiplications across $D$ dimensions —
only integer index lookups into occurrence lists and scatter-adds of precomputed weights. The sparsity bounds *how many* such operations occur; the binarity makes each operation a single index/add. Together they reduce per-frame work from "compare against the whole template" to "follow $a$ short pointer lists."

### 7.4 Per-frame, per-template cost summary

$$
\underbrace{O\!\Big(\textstyle\sum_{d \in Q_t} |P^{(k)}_d|\Big)}_{\text{emission (sparse voting)}}
\;+\;
\underbrace{O(J \cdot T_k)}_{\text{transition (small convolution)}}.
$$

The emission term scales with *matched content*, not with $D$ and not with the number of templates that share no active features. This is the inverted-index retrieval economy, and it is what will allow the method to scale to large databases (§8).

---

## 8. Scaling to many templates (preview)

The per-template description above is the clearest way to define the criterion,
but it is not how one would run it for, say, $K = 1000$ templates. The key observation: a query frame's active features vote only into templates that actually contain those features. By building a **global inverted index** mapping 
$$
d \;\longmapsto\; \{(k, p) : d \in A^{(k)}_p\},
$$

a single active feature deposits its votes simultaneously across all templates where it occurs, and templates sharing no active features with the query receive no work at all. The per-frame cost then scales with the total matched (feature, template, position) incidences — i.e. with *how much content the query shares with the database* — rather than with $K$ directly. Combined with a cheap pre-filter to maintain only a candidate set of progress distributions, this keeps the streaming cost low even for large $K$. The full-distribution forward pass is run only for the surviving candidates. (Details in the scaling note.)

---

## 9. Parameters

All parameters are interpretable and proved insensitive across the tested corruption regimes; none requires per-dataset tuning.

| Parameter | Role | Typical |
|---|---|---|
| $\mu$ (advance mean) | expected positions advanced per query frame | $1.0$ |
| $\sigma$ (advance spread) | how far a single frame may jump (warp tolerance) | $2.0$ |
| $\rho$ (stay probability) | mass kept in place (slow speech / holds) | $0.4$ |
| $\epsilon$ (emission floor) | minimum per-position likelihood (noise robustness) | $0.05$ |
| $\eta$ (uniform leak) | mixing for recovery from a wrong lock | $0.02$ |

The transition kernel is fully determined by $(\mu, \sigma, \rho)$ and is precomputed once.

---

## 10. Algorithm (reference)

**Offline (once):** build the inverted index $P^{(k)}_d$ for every template $k$
and feature $d$; precompute the forward kernel $\boldsymbol{w}$ from $(\mu, \sigma, \rho)$.

**Online (per query frame $t$, per template $k$):**

1. **Emission (sparse voting).** For each active feature $d \in Q_t$, add    $1/|P^{(k)}_d|$ to $e_t[p]$ for every $p \in P^{(k)}_d$. Track whether any    feature matched (informativeness).
2. If informative: normalize $\boldsymbol{e}_t$, add floor $\epsilon/T_k$,
   renormalize. Else: set $\boldsymbol{e}_t = \tfrac{1}{T_k}\mathbf 1$.
3. **Transition.** $\tilde\alpha \leftarrow \boldsymbol{w} * \alpha$ (forward    smear); then $\tilde\alpha \leftarrow (1-\eta)\tilde\alpha + \tfrac{\eta}{T_k}\mathbf 1$.
4. **Combine.** $\hat\alpha \leftarrow \tilde\alpha \odot \boldsymbol{e}_t$;
   $m_t \leftarrow \sum_p \hat\alpha[p]$; $\alpha \leftarrow \hat\alpha / m_t$.
5. **Evidence.** If informative: $\ell_t \leftarrow \log(m_t T_k)$; accumulate    into the running mean $S^{(k)}$ over informative frames.

**Readout (any time):** $\hat k = \arg\max_k S^{(k)}$; margin $=$ gap to the runner-up. The score is available continuously and updates in $O(a)$-scale work per template per frame.

---

## 11. Properties, summarized

| Property | Mechanism |
|---|---|
| Warp tolerance | forward kernel absorbs variable advance rate |
| Acquisition-free | uniform prior + no alignment window |
| Silence-robust | uninformative frames excluded from the score average |
| Noise tolerance | emission floor + recurrence-normalized voting |
| Order discrimination | monotone transition rewards coherent forward paths |
| Efficiency | inverted-index sparse voting; cost scales with matched content |
| No weight tuning | a single evidence criterion, not a weighted sum of terms |

## 12. Known limitation

The criterion's discrimination comes from monotone order. Therefore corruptions that genuinely **destroy order** — for example heavy substitution of the underlying symbols — degrade it toward chance, as they should; it is invariant to nuisances that *preserve* order (rate, onset, silence, sparse bit noise), not to those that *break* it. Strongly periodic or repetitive content is the case to watch, since repeated structure can make an incorrect template's positions advance spuriously; the recurrence-normalized voting mitigates but does not eliminate this, and it is the natural target for empirical stress-testing.