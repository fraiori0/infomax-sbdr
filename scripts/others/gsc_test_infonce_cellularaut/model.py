"""
model.py
--------
Sparse Binary Distributed (SBD) modular architecture for temporal sequence
modelling, targeting the Google Speech Commands dataset.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
Each layer is a pair (CAInterface, SBDEncoder):

  z^{l-1}(t) ∈ {0,1}^C    ← binary sparse output of previous encoder
       │
  ┌────▼────────────────────────────────┐
  │  CAInterface (fixed, non-learned)   │
  │  h^l(t) = AND-of-ORs rule          │
  │  ζ^l(t) = [ z^{l-1}(t) | h^l(t) ] │
  └────────────────────────┬────────────┘
                           │  ζ^l(t) ∈ {0,1}^{C + C_h}
  ┌────────────────────────▼────────────┐
  │  SBDEncoder (learned, InfoNCE)      │
  │  a^l(t) = LayerNorm(W·ζ + b)       │
  │  z^l(t) = (a^l(t) > 0)            │
  └─────────────────────────────────────┘

Training is greedy and layer-wise: each SBDModule is trained with an InfoNCE
loss (implemented in train.py) while all previous modules are frozen.  The
hard threshold creates a natural gradient barrier between layers.

The first module is special: it takes the continuous log-mel spectrogram
(no binary input, no CA interface) and produces the first binary encoding.

CA update rule (AND-of-ORs, analytically sparsity-preserving at k_i = k_h = 3,
p ≈ 0.15):
    h_i(t) = [∨_{j ∈ N_i^input}  z_j(t)]  ∧  [∨_{j ∈ N_i^hidden} h_j(t-1)]

Module conventions
~~~~~~~~~~~~~~~~~~
* All modules accept inputs with *arbitrary leading batch dimensions*.
* Temporal sequences are last-two dimensions: (…, T, features).
* The dataset returns (…, n_mels, T); SBDStack transposes at entry.
* bool tensors carry binary representations (no gradient).
* float tensors carry pre-activations (gradient-bearing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SBDConfig:
    """
    Hyper-parameters for the full SBD stack.

    Parameters
    ----------
    n_input : int
        Dimensionality of the continuous input features (n_mels = 40).
    n_channels : int
        Number of channels C in every encoder layer.
    n_hidden : int
        Number of CA hidden units C_h per layer.
    n_layers : int
        Total number of modules (including the first continuous-input module).
    k_i : int
        Fan-in from input population to each CA hidden unit.
    k_h : int
        Fan-in from hidden population to each CA hidden unit.
    topology : str
        Hidden-to-hidden connectivity.  One of:
        ``"small_world"``, ``"fixed_fanin"``.
    sw_beta : float
        Watts-Strogatz rewiring probability (used only when
        ``topology == "small_world"``).
    layer_norm : bool
        Whether to apply LayerNorm before the hard threshold in each encoder.
    seed : int or None
        Global seed for all connectivity generators.  ``None`` → non-
        deterministic (fresh randomness each run).
    init_sparsity : float
        Initial Bernoulli probability for the CA hidden state at the start
        of each sequence.  Should match the target encoder output sparsity.
        Default ``0.15``.
    """
    n_input:    int   = 40
    n_channels: int   = 128
    n_hidden:   int   = 128
    n_layers:   int   = 5
    k_i:        int   = 3
    k_h:        int   = 3
    topology:   str   = "small_world"
    sw_beta:    float = 0.1
    layer_norm: bool  = True
    seed:          Optional[int] = 42
    init_sparsity: float         = 0.15


# ---------------------------------------------------------------------------
# Connectivity generators
# ---------------------------------------------------------------------------

def _make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Return a seeded Generator, or None (global RNG) when seed is None."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_fixed_fanin_bipartite(
    n_out:     int,
    n_in:      int,
    k:         int,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Binary connectivity matrix for a bipartite graph where every output node
    (hidden unit) receives exactly *k* inputs from distinct input nodes.

    Parameters
    ----------
    n_out : int
        Number of output (hidden) nodes.
    n_in : int
        Number of input nodes.
    k : int
        Exact fan-in per output node.  Must satisfy ``k ≤ n_in``.
    generator : torch.Generator, optional

    Returns
    -------
    A : BoolTensor, shape (n_out, n_in)
        ``A[i, j] = True`` iff hidden unit *i* receives input from unit *j*.
    """
    if k > n_in:
        raise ValueError(
            f"Fan-in k={k} exceeds n_in={n_in}. "
            "Reduce k or increase n_in."
        )
    A = torch.zeros(n_out, n_in, dtype=torch.bool)
    for i in range(n_out):
        cols = torch.randperm(n_in, generator=generator)[:k]
        A[i, cols] = True
    return A


def make_fixed_fanin_graph(
    n:         int,
    k:         int,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Directed random graph where every node has exactly *k* incoming edges from
    distinct, non-self nodes.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Exact in-degree per node.  Must satisfy ``k ≤ n - 1``.
    generator : torch.Generator, optional

    Returns
    -------
    A : BoolTensor, shape (n, n)
        ``A[i, j] = True`` iff node *i* receives input from node *j*.
    """
    if k > n - 1:
        raise ValueError(
            f"Fan-in k={k} exceeds n-1={n-1} (self-loops not allowed)."
        )
    A = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        # Sample k distinct neighbours, excluding self.
        pool = torch.cat([torch.arange(0, i), torch.arange(i + 1, n)])
        perm = torch.randperm(len(pool), generator=generator)[:k]
        A[i, pool[perm]] = True
    return A


def make_watts_strogatz_graph(
    n:         int,
    k:         int,
    beta:      float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Directed Watts-Strogatz small-world graph with exact in-degree *k*.

    Construction
    ~~~~~~~~~~~~
    1. Build a directed ring where each node *i* receives inputs from the
       ``k // 2`` predecessors and ``k // 2`` successors (modular indices),
       giving ``2 * (k // 2)`` ring edges per node.
    2. If *k* is odd, add one extra random non-self, non-existing incoming
       edge per node so that every node has exactly *k* incoming edges before
       rewiring.  This preserves the exact-fan-in guarantee for any *k >= 2*.
    3. For every edge (j -> i), with probability *beta*, rewire the source to
       a uniformly random non-self, non-existing node.

    The result preserves exact in-degree *k* per node at every step, giving
    predictable sparsity in the AND-of-ORs rule for both even and odd *k*.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Exact in-degree per node (``k >= 2``, ``k < n``).
        Both even and odd values are supported.
    beta : float
        Rewiring probability in ``[0, 1]``.  ``0`` -> pure ring / lattice;
        ``1`` -> fully random fixed-fan-in graph.
    generator : torch.Generator, optional

    Returns
    -------
    A : BoolTensor, shape (n, n)
        ``A[i, j] = True`` iff node *i* receives input from node *j*.
    """
    if k < 2:
        raise ValueError(f"k={k} must be at least 2.")
    if k >= n:
        raise ValueError(f"k={k} must be strictly less than n={n}.")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta={beta} must be in [0, 1].")

    half_k = k // 2
    A = torch.zeros(n, n, dtype=torch.bool)

    # ---- Step 1: symmetric ring (floor(k/2) predecessors + successors) -----
    for i in range(n):
        for delta in range(1, half_k + 1):
            A[i, (i - delta) % n] = True   # predecessor
            A[i, (i + delta) % n] = True   # successor

    # ---- Step 2: one extra random edge when k is odd -----------------------
    # Each node currently has 2 * half_k = k - 1 incoming edges; add one more
    # chosen uniformly at random from non-self, non-existing candidates.
    if k % 2 == 1:
        for i in range(n):
            current = set(A[i].nonzero(as_tuple=False).squeeze(1).tolist())
            candidates = [v for v in range(n) if v != i and v not in current]
            # candidates is non-empty because k < n guarantees spare slots.
            idx = torch.randint(
                len(candidates), (), generator=generator
            ).item()
            A[i, candidates[idx]] = True

    # ---- Step 3: rewire edges arriving at each node ------------------------
    for i in range(n):
        sources = A[i].nonzero(as_tuple=False).squeeze(1).tolist()
        for j in sources:
            rnd = torch.rand((), generator=generator).item()
            if rnd >= beta:
                continue
            current = set(A[i].nonzero(as_tuple=False).squeeze(1).tolist())
            candidates = [v for v in range(n) if v != i and v not in current]
            if not candidates:
                continue  # fully connected; cannot rewire
            idx = torch.randint(
                len(candidates), (), generator=generator
            ).item()
            new_j = candidates[idx]
            A[i, j]     = False
            A[i, new_j] = True

    return A

def ca_and_or_step(
    z_t:       Tensor,
    h_prev:    Tensor,
    A_input_f: Tensor,
    A_hidden_f: Tensor,
) -> Tensor:
    """
    Single time-step AND-of-ORs cellular automaton update.

    For hidden unit *i*:
        h_i(t) = [∨_{j ∈ N_i^in}  z_j(t)]  ∧  [∨_{j ∈ N_i^hid} h_j(t-1)]

    Implemented as two thresholded matrix multiplications followed by a
    boolean AND.  Handles arbitrary leading batch dimensions via broadcasting.

    Parameters
    ----------
    z_t : BoolTensor, shape (…, C_in)
        Current input population (binary sparse, from encoder output).
    h_prev : BoolTensor, shape (…, C_h)
        Previous hidden state.
    A_input_f : FloatTensor, shape (C_h, C_in)
        Input connectivity matrix (float for matmul efficiency).
    A_hidden_f : FloatTensor, shape (C_h, C_h)
        Hidden connectivity matrix (float for matmul efficiency).

    Returns
    -------
    h_new : BoolTensor, shape (…, C_h)
    """
    # (…, C_h): unit i is 1 iff at least one input neighbour is active.
    input_or = (z_t.float() @ A_input_f.t()) > 0

    # (…, C_h): unit i is 1 iff at least one hidden neighbour was active.
    hidden_or = (h_prev.float() @ A_hidden_f.t()) > 0

    return input_or & hidden_or   # (…, C_h)


# ---------------------------------------------------------------------------
# CA Interface module
# ---------------------------------------------------------------------------

class CAInterface(nn.Module):
    """
    Fixed (non-learned) cellular automaton interface between encoder layers.

    At each time step *t*, the hidden population is updated by the AND-of-ORs
    rule driven by the current binary encoder output ``z(t)``.  The interface
    vector for the next encoder is the concatenation of the input and hidden
    populations across the full time axis.

    Sparsity preservation
    ~~~~~~~~~~~~~~~~~~~~~
    With fan-in ``k_i = k_h = 3`` and input sparsity ``p ≈ 0.15`` the
    analytical expected output sparsity is:

        P(h_i = 1) = [1 − (1−p)^k_i] × [1 − (1−p)^k_h]
                   = [1 − 0.85^3]^2  ≈  0.149  ≈  p

    No kWTA or running statistics are required.

    Parameters
    ----------
    C_in : int
        Dimensionality of the input population (= encoder output channels).
    C_h : int
        Number of hidden units.
    k_i : int
        Fan-in from input population to each hidden unit.
    k_h : int
        Fan-in from hidden population to each hidden unit.
    topology : str
        ``"small_world"`` or ``"fixed_fanin"``.
    sw_beta : float
        Watts-Strogatz rewiring probability (ignored for ``"fixed_fanin"``).
    seed : int, optional
        Connectivity seed for reproducibility.
    init_sparsity : float
        Bernoulli probability for hidden unit activation at sequence start.
        Default ``0.15``.  Must be > 0 to avoid the all-zero deadlock.
    """

    def __init__(
        self,
        C_in:          int,
        C_h:           int,
        k_i:           int   = 3,
        k_h:           int   = 3,
        topology:      str   = "small_world",
        sw_beta:       float = 0.1,
        seed:          Optional[int] = None,
        init_sparsity: float = 0.15,
    ) -> None:
        super().__init__()

        self.C_in          = C_in
        self.C_h           = C_h
        self.init_sparsity = init_sparsity

        gen = _make_generator(seed)

        # --- Input connectivity: (C_h, C_in) --------------------------------
        A_input = make_fixed_fanin_bipartite(C_h, C_in, k_i, generator=gen)

        # --- Hidden connectivity: (C_h, C_h) --------------------------------
        if topology == "small_world":
            A_hidden = make_watts_strogatz_graph(C_h, k_h, sw_beta,
                                                 generator=gen)
        elif topology == "fixed_fanin":
            A_hidden = make_fixed_fanin_graph(C_h, k_h, generator=gen)
        else:
            raise ValueError(
                f"Unknown topology '{topology}'. "
                "Choose 'small_world' or 'fixed_fanin'."
            )

        # Register as buffers: moved to the right device with model.to(),
        # included in state_dict(), but not updated by the optimiser.
        # Float32 copies are kept for efficient matmul; bool copies for
        # serialisation and inspection.
        self.register_buffer("A_input_b",  A_input)
        self.register_buffer("A_hidden_b", A_hidden)
        self.register_buffer("A_input_f",  A_input.float())
        self.register_buffer("A_hidden_f", A_hidden.float())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def C_out(self) -> int:
        """Width of the interface vector: C_in + C_h."""
        return self.C_in + self.C_h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: Tensor) -> Tensor:
        """
        Run the CA over the time axis and return the interface tensor.

        Parameters
        ----------
        z : BoolTensor, shape (…, T, C_in)
            Binary sparse encoder output for an entire sequence.

        Returns
        -------
        zeta : BoolTensor, shape (…, T, C_in + C_h)
            Concatenation of input and hidden populations at every time step.
        """
        if z.dtype != torch.bool:
            raise TypeError(
                f"CAInterface expects a BoolTensor; got {z.dtype}. "
                "Pass z = (a > 0) before calling the CA."
            )

        *batch_dims, T, C_in = z.shape
        if C_in != self.C_in:
            raise ValueError(
                f"Input channel mismatch: expected {self.C_in}, got {C_in}."
            )

        # Initialise hidden state with Bernoulli(init_sparsity).
        # All-zero initialisation causes a permanent deadlock: the AND gate
        # requires hidden_or > 0, which requires h_prev > 0, which requires
        # the AND gate to have fired — a circular dependency with no escape.
        # Bernoulli(p) matches the analytical steady-state distribution and
        # allows the CA to produce meaningful activity from the first frame.
        h = torch.bernoulli(
            torch.full(
                (*batch_dims, self.C_h),
                self.init_sparsity,
                device=z.device,
            )
        ).bool()                                           # (…, C_h) bool

        # Pre-allocate output buffer for the hidden sequence.
        h_seq = z.new_zeros(*batch_dims, T, self.C_h)     # (…, T, C_h) bool

        for t in range(T):
            h = ca_and_or_step(
                z[..., t, :],
                h,
                self.A_input_f,
                self.A_hidden_f,
            )                                              # (…, C_h) bool
            h_seq[..., t, :] = h

        # Concatenate along the feature axis.
        return torch.cat([z, h_seq], dim=-1)               # (…, T, C_in+C_h)

    def extra_repr(self) -> str:
        avg_in  = self.A_input_f.sum(dim=1).mean().item()
        avg_hid = self.A_hidden_f.sum(dim=1).mean().item()
        p_out   = (
            (1 - (1 - 0.15) ** avg_in) *
            (1 - (1 - 0.15) ** avg_hid)
        )
        return (
            f"C_in={self.C_in}, C_h={self.C_h}, "
            f"avg_k_i={avg_in:.1f}, avg_k_h={avg_hid:.1f}, "
            f"predicted_p_out≈{p_out:.3f}"
        )


# ---------------------------------------------------------------------------
# Single-layer encoder
# ---------------------------------------------------------------------------

class SBDEncoder(nn.Module):
    """
    Single linear layer followed by optional LayerNorm and a hard threshold.

    The encoder is the *only* learned component in each SBDModule.  It maps
    a continuous (or binary-as-float) interface vector to a binary sparse
    representation.

    The forward pass returns *both* the continuous pre-activation ``a`` (needed
    by the InfoNCE critic during training) and the binary output ``z`` (needed
    by the next layer at all times).

    Parameters
    ----------
    C_in : int
        Input dimensionality.
    C_out : int
        Output dimensionality (number of binary units).
    layer_norm : bool
        If ``True``, apply ``nn.LayerNorm(C_out)`` before the threshold.
        Strongly recommended for threshold stability.
    """

    def __init__(
        self,
        C_in:       int,
        C_out:      int,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(C_in, C_out)
        self.norm   = nn.LayerNorm(C_out) if layer_norm else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform for the linear weight; zero bias."""
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (…, C_in)
            Continuous (spectrogram) or boolean (binary sparse) input.
            Boolean inputs are converted to float32 internally.

        Returns
        -------
        a : FloatTensor, shape (…, C_out)
            Normalised pre-activation.  This is the gradient-bearing tensor
            used by the InfoNCE critic; requires_grad follows the usual
            PyTorch autograd rules.
        z : BoolTensor, shape (…, C_out)
            Hard-thresholded binary output.  No gradient.
        """
        # Ensure float input for the linear layer.
        x_f = x.float() if x.dtype != torch.float32 else x

        a = self.norm(self.linear(x_f))   # (…, C_out) float, grad-bearing
        z = (a > 0)                        # (…, C_out) bool, no grad
        return a, z

    def extra_repr(self) -> str:
        return (
            f"C_in={self.linear.in_features}, "
            f"C_out={self.linear.out_features}, "
            f"layer_norm={isinstance(self.norm, nn.LayerNorm)}"
        )


# ---------------------------------------------------------------------------
# Single SBD module (CA interface + encoder)
# ---------------------------------------------------------------------------

class SBDModule(nn.Module):
    """
    One stage of the SBD stack: CA interface followed by SBDEncoder.

    The first module in the stack is constructed with ``ca=None`` and takes
    a *continuous* input (log-mel spectrogram values).  All subsequent modules
    take *binary* inputs and use the CA interface.

    Parameters
    ----------
    encoder : SBDEncoder
        The learned encoder for this module.
    ca : CAInterface or None
        Pre-built CA interface.  ``None`` for the first module.
    """

    def __init__(
        self,
        encoder: SBDEncoder,
        ca:      Optional[CAInterface] = None,
    ) -> None:
        super().__init__()

        self.ca      = ca        # may be None
        self.encoder = encoder

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def make_first(
        cls,
        n_input:    int,
        n_channels: int,
        layer_norm: bool = True,
    ) -> "SBDModule":
        """
        Build the first module (continuous input, no CA).

        Parameters
        ----------
        n_input : int
            Number of input features (n_mels).
        n_channels : int
            Number of binary output channels.
        layer_norm : bool
        """
        encoder = SBDEncoder(n_input, n_channels, layer_norm=layer_norm)
        return cls(encoder=encoder, ca=None)

    @classmethod
    def make_subsequent(
        cls,
        n_channels:    int,
        n_hidden:      int,
        layer_norm:    bool  = True,
        k_i:           int   = 3,
        k_h:           int   = 3,
        topology:      str   = "small_world",
        sw_beta:       float = 0.1,
        seed:          Optional[int] = None,
        init_sparsity: float = 0.15,
    ) -> "SBDModule":
        """
        Build a non-first module (binary input, full CA interface).

        Parameters
        ----------
        n_channels : int
            Number of binary channels in (= out from previous layer).
        n_hidden : int
            Number of CA hidden units.
        layer_norm, k_i, k_h, topology, sw_beta, seed, init_sparsity :
            Forwarded to :class:`CAInterface` / :class:`SBDEncoder`.
        """
        ca = CAInterface(
            C_in=n_channels,
            C_h=n_hidden,
            k_i=k_i,
            k_h=k_h,
            topology=topology,
            sw_beta=sw_beta,
            seed=seed,
            init_sparsity=init_sparsity,
        )
        encoder = SBDEncoder(
            C_in=n_channels + n_hidden,
            C_out=n_channels,
            layer_norm=layer_norm,
        )
        return cls(encoder=encoder, ca=ca)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (…, T, C_in)
            For the first module: float32 spectrogram, (…, T, n_mels).
            For subsequent modules: bool binary sparse, (…, T, C).

        Returns
        -------
        a : FloatTensor, shape (…, T, C_out)
            Pre-activation (for InfoNCE critic).
        z : BoolTensor, shape (…, T, C_out)
            Binary sparse output (for next module / classifier).
        """
        if self.ca is not None:
            # Binary path: run CA to get temporal context, then encode.
            zeta = self.ca(x)           # (…, T, C + C_h) bool
        else:
            # First module: pass spectrogram directly to the encoder.
            zeta = x                    # (…, T, n_mels) float

        a, z = self.encoder(zeta)       # (…, T, C_out)
        return a, z


# ---------------------------------------------------------------------------
# Full SBD stack
# ---------------------------------------------------------------------------

class SBDStack(nn.Module):
    """
    Stack of :class:`SBDModule` instances trained greedily with InfoNCE.

    Input convention
    ~~~~~~~~~~~~~~~~
    The stack expects spectrograms in the dataset format ``(…, n_mels, T)``.
    It transposes the last two dimensions internally before processing, so
    that all modules see ``(…, T, features)``.

    Greedy training API
    ~~~~~~~~~~~~~~~~~~~
    Use :meth:`freeze_up_to` to freeze all modules with index ≤ ``layer_idx``
    before training module ``layer_idx + 1``.  Call :meth:`unfreeze_all` to
    restore all parameters to trainable state.

    Parameters
    ----------
    cfg : SBDConfig
        Architecture hyper-parameters.
    """

    def __init__(self, cfg: SBDConfig) -> None:
        super().__init__()

        self.cfg = cfg
        layers: List[SBDModule] = []

        # --- Module 0: continuous input, no CA ----------------------------
        layers.append(
            SBDModule.make_first(
                n_input=cfg.n_input,
                n_channels=cfg.n_channels,
                layer_norm=cfg.layer_norm,
            )
        )

        # --- Modules 1 … n_layers-1: binary input, with CA ---------------
        # Each layer gets a deterministic but distinct seed derived from the
        # global seed so that connectivity is reproducible per layer.
        for l in range(1, cfg.n_layers):
            layer_seed = (
                (cfg.seed * 1_000 + l) if cfg.seed is not None else None
            )
            layers.append(
                SBDModule.make_subsequent(
                    n_channels=cfg.n_channels,
                    n_hidden=cfg.n_hidden,
                    layer_norm=cfg.layer_norm,
                    k_i=cfg.k_i,
                    k_h=cfg.k_h,
                    topology=cfg.topology,
                    sw_beta=cfg.sw_beta,
                    seed=layer_seed,
                    init_sparsity=cfg.init_sparsity,
                )
            )

        # nn.ModuleList ensures proper device movement and parameter tracking.
        # Note: `self.modules` is a reserved nn.Module method, so we use
        # `self.layers` as the attribute name.
        self.layers = nn.ModuleList(layers)

    # ------------------------------------------------------------------
    # Greedy training helpers
    # ------------------------------------------------------------------

    def freeze_up_to(self, layer_idx: int) -> None:
        """
        Freeze parameters of modules 0 … *layer_idx* (inclusive).

        Should be called before training module ``layer_idx + 1``.

        Parameters
        ----------
        layer_idx : int
            Last module index to freeze.  Clamped to ``[0, n_layers - 1]``.
        """
        for i in range(min(layer_idx + 1, len(self.layers))):
            for p in self.layers[i].parameters():
                p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Restore all module parameters to trainable state."""
        for p in self.parameters():
            p.requires_grad_(True)

    def set_layer_trainable(self, layer_idx: int) -> None:
        """
        Convenience: freeze all layers except *layer_idx*.

        Typical greedy training loop::

            for l in range(cfg.n_layers):
                stack.set_layer_trainable(l)
                optimizer = torch.optim.AdamW(
                    stack.layers[l].encoder.parameters(), ...
                )
                # train ...
        """
        # First freeze everything, then unfreeze the target layer's encoder.
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.layers[layer_idx].encoder.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        spec:      Tensor,
        up_to:     int = -1,
        no_grad_below: bool = False,
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Run the stack and return intermediate (a, z) pairs.

        Parameters
        ----------
        spec : FloatTensor, shape (…, n_mels, T)
            Log-mel spectrogram in dataset format.
        up_to : int
            If non-negative, stop after (and including) module *up_to*.
            Default ``-1`` runs all modules.
        no_grad_below : bool
            If ``True``, all modules *before* the last executed module are
            run under ``torch.no_grad()``.  Useful during greedy training to
            save memory on intermediate activations.  The last module always
            runs with the current grad mode.

        Returns
        -------
        outputs : list of (a, z) pairs, one per executed module.
            ``a`` — FloatTensor (…, T, C_out), pre-activation.
            ``z`` — BoolTensor  (…, T, C_out), binary sparse output.
        """
        # Transpose from (…, n_mels, T) → (…, T, n_mels) for sequential
        # processing along the time axis.
        x = spec.transpose(-1, -2)   # (…, T, n_mels) float

        n = len(self.layers) if up_to < 0 else (up_to + 1)
        n = min(n, len(self.layers))

        outputs: List[Tuple[Tensor, Tensor]] = []

        for i, module in enumerate(self.layers[:n]):
            is_last = (i == n - 1)
            if no_grad_below and not is_last:
                with torch.no_grad():
                    a, z = module(x)
            else:
                a, z = module(x)
            outputs.append((a, z))
            # Pass binary output as input to the next layer.
            # (Detach from graph for all but the current training layer.)
            x = z.detach() if (no_grad_below and not is_last) else z

        return outputs

    def get_binary_representations(
        self,
        spec:   Tensor,
        layers: Optional[List[int]] = None,
    ) -> List[Tensor]:
        """
        Return binary sparse output tensors for the requested layers.

        Convenience wrapper around :meth:`forward` that runs under
        ``torch.no_grad()`` and returns only the ``z`` tensors.

        Parameters
        ----------
        spec : FloatTensor, shape (…, n_mels, T)
        layers : list of int, optional
            Layer indices to return.  Default: all layers.

        Returns
        -------
        zs : list of BoolTensor, shape (…, T, C)
        """
        idx_set = set(range(len(self.layers))) if layers is None else set(layers)
        max_layer = max(idx_set)

        with torch.no_grad():
            outputs = self.forward(spec, up_to=max_layer)

        return [outputs[i][1] for i in sorted(idx_set) if i < len(outputs)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_channels(self) -> int:
        return self.cfg.n_channels


# ---------------------------------------------------------------------------
# Linear classifier head
# ---------------------------------------------------------------------------

class LinearClassifier(nn.Module):
    """
    Linear probe for evaluating frozen SBD representations.

    Takes the global-average-pooled binary representation from one (or more)
    SBD layers and produces class logits.

    Parameters
    ----------
    n_channels : int
        Number of binary channels per selected layer.
    n_classes : int
        Number of target classes (35 for GSC v2).
    n_probe_layers : int
        Number of SBD layers whose representations are concatenated before
        the linear transform.  Default ``1`` (last layer only).
    """

    def __init__(
        self,
        n_channels:     int,
        n_classes:      int,
        n_probe_layers: int = 1,
    ) -> None:
        super().__init__()

        self.n_probe_layers = n_probe_layers
        self.linear = nn.Linear(n_channels * n_probe_layers, n_classes)

    def forward(self, zs: List[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        zs : list of BoolTensor, shape (…, T, C)
            Binary representations from *n_probe_layers* SBD layers.

        Returns
        -------
        logits : FloatTensor, shape (…, n_classes)
        """
        if len(zs) != self.n_probe_layers:
            raise ValueError(
                f"Expected {self.n_probe_layers} tensors, got {len(zs)}."
            )

        # Global average pool over the time axis, then concatenate channels.
        pooled = [z.float().mean(dim=-2) for z in zs]   # each (…, C)
        combined = torch.cat(pooled, dim=-1)             # (…, C * n_layers)
        return self.linear(combined)                     # (…, n_classes)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_model(cfg: SBDConfig) -> Tuple[SBDStack, LinearClassifier]:
    """
    Instantiate and return the full (stack, classifier) pair from a config.

    Parameters
    ----------
    cfg : SBDConfig

    Returns
    -------
    stack : SBDStack
    classifier : LinearClassifier
        Single-layer probe on the final SBD layer.
    """
    stack      = SBDStack(cfg)
    classifier = LinearClassifier(
        n_channels=cfg.n_channels,
        n_classes=35,
        n_probe_layers=1,
    )
    return stack, classifier