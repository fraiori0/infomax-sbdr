"""
dataset.py
----------
Google Speech Commands v2 (GSC) dataset with log-mel spectrogram preprocessing.

Preprocessing pipeline
~~~~~~~~~~~~~~~~~~~~~~
1. Load .wav, mix to mono, resample to 16 kHz if needed
2. Pad / trim to exactly 16 000 samples (1 second)
3. Log-mel spectrogram: 40 mel bins, 25 ms window (n_fft=400), 10 ms hop
   → 98 frames per utterance
4. Per-utterance CMVN normalisation (zero-mean, unit-variance per mel bin
   over the time axis)
5. Optional SpecAugment at __getitem__ time (training only)

Design decisions
~~~~~~~~~~~~~~~~
* Steps 1–4 are pre-computed and cached in RAM as a (N, n_mels, n_frames)
  float32 tensor when precompute=True (default).  For the full training split
  (~85 k clips) this requires ≈ 1.3 GB; practical on any modern machine.
* CMVN is applied *inside* the cache so normalised features are stored,
  keeping __getitem__ allocation-free in the common case.
* SpecAugment lives in a separate callable so it can be swapped or removed
  without rebuilding the cache.
* All transforms are stateless callables that respect arbitrary leading batch
  dimensions (…, n_mels, n_frames) / (…, n_samples) wherever applicable.

Usage example
~~~~~~~~~~~~~
    from dataset import GSCDataset, SpecAugmentTransform

    GSC_ROOT = "/path/to/speech_commands_v0.02"

    train_ds = GSCDataset(
        root=GSC_ROOT,
        split="train",
        precompute=True,
        augment=SpecAugmentTransform(),
    )
    val_ds = GSCDataset(root=GSC_ROOT, split="val")
    test_ds = GSCDataset(root=GSC_ROOT, split="test")

    loader = torch.utils.data.DataLoader(train_ds, batch_size=256,
                                         shuffle=True, num_workers=4,
                                         pin_memory=True)
    spec, label = next(iter(loader))   # (256, 40, 98),  (256,)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Native GSC sample rate — files are already 16 kHz; kept explicit for safety.
SAMPLE_RATE: int = 16_000

#: Number of mel frequency bins.
N_MELS: int = 40

#: FFT window length in samples (25 ms × 16 kHz).
N_FFT: int = 400

#: Hop length in samples (10 ms × 16 kHz).
HOP_LENGTH: int = 160

#: Lowest mel filter frequency.
F_MIN: float = 20.0

#: Highest mel filter frequency (= Nyquist at 16 kHz).
F_MAX: float = 8_000.0

#: Clip length in samples (1 second).
TARGET_SAMPLES: int = 16_000

#: Expected frame count after padding to TARGET_SAMPLES.
#  = 1 + (16000 − 400) // 160  =  98
N_FRAMES: int = 98

#: Small constant added before log to prevent log(0) and influence sparsity.
LOG_EPS: float = 1e-6

# ---------------------------------------------------------------------------
# GSC v2 class registry (35 words, alphabetically sorted)
# ---------------------------------------------------------------------------

GSC_CLASSES: List[str] = [
    "backward", "bed",    "bird",   "cat",    "dog",
    "down",     "eight",  "five",   "follow", "forward",
    "four",     "go",     "happy",  "house",  "learn",
    "left",     "marvin", "nine",   "no",     "off",
    "on",       "one",    "right",  "seven",  "sheila",
    "six",      "stop",   "three",  "tree",   "two",
    "up",       "visual", "wow",    "yes",    "zero",
]

CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(GSC_CLASSES)}

# ---------------------------------------------------------------------------
# Transform components
# ---------------------------------------------------------------------------


class LogMelTransform:
    """
    Raw waveform → log-mel spectrogram.

    Accepts tensors with arbitrary leading batch dimensions.

    Parameters
    ----------
    sample_rate : int
        Input sample rate (Hz).  Default: 16 000.
    n_fft : int
        FFT window length in samples.  Default: 400 (25 ms).
    hop_length : int
        Hop length in samples.  Default: 160 (10 ms).
    n_mels : int
        Number of mel filter bins.  Default: 40.
    f_min, f_max : float
        Frequency range for mel filters.
    log_eps : float
        Stabilisation constant before the logarithm; also controls emergent
        sparsity in downstream binary representations.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft:       int = N_FFT,
        hop_length:  int = HOP_LENGTH,
        n_mels:      int = N_MELS,
        f_min:     float = F_MIN,
        f_max:     float = F_MAX,
        log_eps:   float = LOG_EPS,
    ) -> None:
        self.log_eps = log_eps
        # torchaudio.transforms.MelSpectrogram is an nn.Module; we wrap it
        # as a plain callable here so the transform is serialisable without
        # needing a device.
        self._mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            # center=False prevents torchaudio from padding the waveform by
            # n_fft // 2 = 200 samples on each side before the STFT.
            # With center=True (the default) a 16 000-sample clip produces
            # 1 + (16400 - 400) // 160 = 101 frames instead of the expected
            # 1 + (16000 - 400) // 160 = 98 frames.
            center=False,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : Tensor, shape (…, T)

        Returns
        -------
        log_mel : Tensor, shape (…, n_mels, n_frames)
        """
        mel = self._mel(waveform)                    # (…, n_mels, n_frames)
        return torch.log(mel + self.log_eps)


class CMVNTransform:
    """
    Per-utterance Cepstral Mean and Variance Normalisation.

    Subtracts the time-axis mean and divides by the time-axis standard
    deviation for each mel bin independently.  Applied after log-mel
    computation.

    Accepts tensors with arbitrary leading batch dimensions.

    Parameters
    ----------
    eps : float
        Small constant added to the denominator for numerical stability.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        log_mel : Tensor, shape (…, n_mels, n_frames)

        Returns
        -------
        normalised : Tensor, same shape.
        """
        # Normalise over the time (last) dimension, per mel bin.
        mean = log_mel.mean(dim=-1, keepdim=True)   # (…, n_mels, 1)
        std  = log_mel.std( dim=-1, keepdim=True)   # (…, n_mels, 1)
        return (log_mel - mean) / (std + self.eps)


class SpecAugmentTransform:
    """
    SpecAugment data augmentation (Park et al., 2019).

    Applies independent time and frequency masks.  Intended for training
    only — pass an instance only to the training :class:`GSCDataset`.

    Parameters chosen conservatively for short single-word utterances:
        2 time masks  × max 15 frames  (≈ 150 ms)
        2 freq masks  × max 5 bins

    Accepts tensors with arbitrary leading batch dimensions.

    Parameters
    ----------
    n_time_masks : int
    max_time_width : int
        Maximum width of each time mask in frames.
    n_freq_masks : int
    max_freq_width : int
        Maximum width of each frequency mask in mel bins.
    """

    def __init__(
        self,
        n_time_masks:   int = 2,
        max_time_width: int = 15,
        n_freq_masks:   int = 2,
        max_freq_width: int = 5,
    ) -> None:
        self.n_time_masks   = n_time_masks
        self.max_time_width = max_time_width
        self.n_freq_masks   = n_freq_masks
        self.max_freq_width = max_freq_width

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spec : Tensor, shape (…, n_mels, n_frames)

        Returns
        -------
        masked : Tensor, same shape.
        """
        out     = spec.clone()
        n_mels  = out.shape[-2]
        n_frames = out.shape[-1]

        for _ in range(self.n_time_masks):
            w = int(torch.randint(0, self.max_time_width + 1, ()).item())
            if w == 0 or n_frames <= w:
                continue
            t = int(torch.randint(0, n_frames - w, ()).item())
            out[..., :, t : t + w] = 0.0

        for _ in range(self.n_freq_masks):
            w = int(torch.randint(0, self.max_freq_width + 1, ()).item())
            if w == 0 or n_mels <= w:
                continue
            f = int(torch.randint(0, n_mels - w, ()).item())
            out[..., f : f + w, :] = 0.0

        return out


# ---------------------------------------------------------------------------
# Waveform I/O helper
# ---------------------------------------------------------------------------


def load_and_pad_waveform(
    path:           str | Path,
    target_samples: int = TARGET_SAMPLES,
    target_sr:      int = SAMPLE_RATE,
) -> torch.Tensor:
    """
    Load a WAV file, mix to mono, resample if necessary, and pad / trim to
    *target_samples* samples.

    Parameters
    ----------
    path : str or Path
        Path to a .wav file.
    target_samples : int
        Desired output length in samples.
    target_sr : int
        Desired sample rate.

    Returns
    -------
    waveform : Tensor, shape (1, target_samples), dtype float32.
    """
    waveform, sr = torchaudio.load(str(path))   # (channels, T)

    # Mix to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample only when needed to avoid unnecessary computation.
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

    # Pad (zero) or trim to exactly target_samples.
    n = waveform.shape[-1]
    if n >= target_samples:
        waveform = waveform[..., :target_samples]
    else:
        waveform = F.pad(waveform, (0, target_samples - n))

    return waveform.float()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GSCDataset(Dataset):
    """
    Google Speech Commands v2 dataset.

    Splits follow the official validation_list.txt / testing_list.txt files
    bundled with the corpus:

    * Files in testing_list.txt     → ``split="test"``
    * Files in validation_list.txt  → ``split="val"``
    * All remaining labelled files  → ``split="train"``

    Each :meth:`__getitem__` returns a ``(spec, label)`` pair where

    * ``spec``  is a float32 tensor of shape ``(n_mels, n_frames)`` containing
      the CMVN-normalised log-mel spectrogram (with optional SpecAugment).
    * ``label`` is an ``int`` in ``[0, num_classes)``.

    Parameters
    ----------
    root : str or Path
        Root directory of the extracted GSC corpus (the folder that contains
        ``validation_list.txt``, ``testing_list.txt``, and the per-class
        subdirectories).
    split : {"train", "val", "test"}
        Which subset to expose.
    precompute : bool
        If ``True`` (default), pre-compute and cache all CMVN-normalised
        log-mel spectrograms at construction time.  Recommended for training;
        eliminates repeated I/O and FFT overhead at the cost of RAM.
    augment : callable, optional
        A callable ``(spec) → spec`` applied at ``__getitem__`` time.
        Pass a :class:`SpecAugmentTransform` instance for the training split.
    log_mel_kwargs : dict, optional
        Keyword arguments forwarded to :class:`LogMelTransform`, e.g.
        ``{"n_mels": 80, "log_eps": 1e-5}`` to override defaults.
    cmvn_eps : float
        Epsilon for the CMVN denominator.  Default: ``1e-8``.
    max_samples : int, optional
        If set, truncate the split to at most *max_samples* items (taken
        from the beginning of the sorted file list).  Useful for rapid
        development and debugging without loading the full corpus.
        ``None`` (default) loads everything.
    cache_dir : str or Path, optional
        Directory in which to persist the pre-computed spectrogram tensor as
        a ``.pt`` file.  On the first run the cache is built and saved; on
        subsequent runs it is loaded directly, reducing startup from ~10 min
        to a few seconds.  The filename encodes the split, ``max_samples``,
        and the key ``LogMelTransform`` parameters so that changing any
        preprocessing hyper-parameter automatically invalidates the old file.
        ``None`` (default) keeps the cache in RAM only (no disk write).
    """

    def __init__(
        self,
        root:            str | Path,
        split:           str = "train",
        precompute:      bool = True,
        augment:         Optional[Callable] = None,
        log_mel_kwargs:  Optional[dict] = None,
        cmvn_eps:        float = 1e-8,
        max_samples:     Optional[int] = None,
        cache_dir:       Optional[str | Path] = None,
    ) -> None:
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test'; got '{split}'"
            )

        self.root        = Path(root)
        self.split       = split
        self.augment     = augment
        self.max_samples = max_samples
        self.cache_dir   = Path(cache_dir) if cache_dir is not None else None

        # --- Build preprocessing pipeline -----------------------------------
        lm_kw = log_mel_kwargs or {}
        self._log_mel = LogMelTransform(**lm_kw)
        self._cmvn    = CMVNTransform(eps=cmvn_eps)

        # --- Collect paths and integer labels for the split -----------------
        self._paths, self._labels = self._collect_split()

        # Truncate for quick-test mode *after* collecting the full split so
        # that the path ordering is always deterministic (sorted per class).
        if max_samples is not None:
            self._paths  = self._paths[:max_samples]
            self._labels = self._labels[:max_samples]

        # --- Optionally pre-compute cache -----------------------------------
        self._cache: Optional[torch.Tensor] = None
        if precompute:
            self._cache = self._load_or_build_cache(lm_kw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_list(self, filename: str) -> frozenset[str]:
        """Return a frozenset of relative paths from a GSC split list file."""
        p = self.root / filename
        if not p.exists():
            raise FileNotFoundError(
                f"Expected split list file not found: {p}\n"
                "Make sure 'root' points to the top-level GSC directory."
            )
        with p.open() as fh:
            return frozenset(line.strip() for line in fh if line.strip())

    def _collect_split(self) -> Tuple[List[Path], List[int]]:
        """
        Walk the per-class directories and assign each .wav file to its split.

        Returns
        -------
        paths  : list of Path objects
        labels : list of int class indices, aligned with paths
        """
        val_set  = self._read_list("validation_list.txt")
        test_set = self._read_list("testing_list.txt")

        paths:  List[Path] = []
        labels: List[int]  = []

        for cls in GSC_CLASSES:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            cls_idx = CLASS_TO_IDX[cls]
            for wav in sorted(cls_dir.glob("*.wav")):
                rel = f"{cls}/{wav.name}"   # matches format in list files
                in_test = rel in test_set
                in_val  = rel in val_set
                if   self.split == "test"  and in_test:
                    paths.append(wav); labels.append(cls_idx)
                elif self.split == "val"   and in_val:
                    paths.append(wav); labels.append(cls_idx)
                elif self.split == "train" and not in_test and not in_val:
                    paths.append(wav); labels.append(cls_idx)

        if not paths:
            raise RuntimeError(
                f"No audio files found for split='{self.split}' under "
                f"'{self.root}'.\nVerify the path and that the corpus is "
                "fully extracted."
            )
        return paths, labels

    def _preprocess(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply LogMel + CMVN to a (1, T) waveform → (n_mels, n_frames)."""
        spec = self._log_mel(wav)   # (1, n_mels, n_frames)
        spec = self._cmvn(spec)     # (1, n_mels, n_frames)
        return spec.squeeze(0)      # (n_mels, n_frames)

    def _cache_path(self, lm_kw: dict) -> Optional[Path]:
        """
        Return the path for the on-disk cache file, or ``None`` if
        ``cache_dir`` was not set.

        The filename encodes all parameters that affect the spectrogram values
        so that a change in any hyper-parameter automatically produces a
        different (new) cache file rather than silently loading a stale one.
        """
        if self.cache_dir is None:
            return None

        # Build a short, human-readable fingerprint of the preprocessing params.
        n_mels     = lm_kw.get("n_mels",      N_MELS)
        n_fft      = lm_kw.get("n_fft",       N_FFT)
        hop_length = lm_kw.get("hop_length",  HOP_LENGTH)
        f_min      = lm_kw.get("f_min",       F_MIN)
        f_max      = lm_kw.get("f_max",       F_MAX)
        log_eps    = lm_kw.get("log_eps",     LOG_EPS)
        ms_tag     = f"_max{self.max_samples}" if self.max_samples else ""

        fname = (
            f"gsc_{self.split}{ms_tag}"
            f"_mels{n_mels}_fft{n_fft}_hop{hop_length}"
            f"_fmin{f_min:.0f}_fmax{f_max:.0f}"
            f"_eps{log_eps:.0e}"
            ".pt"
        )
        return self.cache_dir / fname

    def _load_or_build_cache(self, lm_kw: dict) -> torch.Tensor:
        """
        Return the spectrogram cache, loading from disk if available and
        valid, otherwise building from scratch (and optionally saving).
        """
        cache_path = self._cache_path(lm_kw)

        if cache_path is not None and cache_path.exists():
            print(
                f"[GSCDataset] Loading pre-computed cache from "
                f"{cache_path} …"
            )
            cache = torch.load(cache_path, weights_only=True)
            # Sanity-check shape matches current split size.
            if cache.shape[0] != len(self._paths):
                print(
                    f"[GSCDataset] WARNING: cached shape {tuple(cache.shape)} "
                    f"does not match current split size {len(self._paths)}. "
                    "Rebuilding …"
                )
                cache = self._build_cache()
                if cache_path is not None:
                    self._save_cache(cache, cache_path)
            else:
                size_gb = cache.nbytes / 1e9
                print(
                    f"[GSCDataset] Cache loaded — shape {tuple(cache.shape)}, "
                    f"{size_gb:.2f} GB"
                )
            return cache

        # No valid on-disk cache — build from scratch.
        cache = self._build_cache()

        if cache_path is not None:
            self._save_cache(cache, cache_path)

        return cache

    @staticmethod
    def _save_cache(cache: torch.Tensor, path: Path) -> None:
        """Persist *cache* to *path*, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, path)
        print(f"[GSCDataset] Cache saved to {path}")

    def _build_cache(self) -> torch.Tensor:
        """
        Pre-compute every log-mel+CMVN spectrogram and pack into a single
        (N, n_mels, n_frames) float32 tensor stored on CPU.
        """
        print(
            f"[GSCDataset] Pre-computing {len(self._paths):,} spectrograms "
            f"for split='{self.split}' …"
        )
        specs = []
        for path in self._paths:
            wav  = load_and_pad_waveform(path)   # (1, TARGET_SAMPLES)
            spec = self._preprocess(wav)          # (n_mels, n_frames)
            specs.append(spec)

        cache = torch.stack(specs, dim=0)         # (N, n_mels, n_frames)
        size_gb = cache.nbytes / 1e9
        print(
            f"[GSCDataset] Cache ready — shape {tuple(cache.shape)}, "
            f"{size_gb:.2f} GB"
        )
        return cache

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns
        -------
        spec  : float32 Tensor, shape (n_mels, n_frames)
                CMVN-normalised log-mel spectrogram.
                SpecAugment is applied if ``self.augment`` is set.
        label : int in [0, num_classes)
        """
        if self._cache is not None:
            spec = self._cache[idx].clone()         # (n_mels, n_frames)
        else:
            wav  = load_and_pad_waveform(self._paths[idx])
            spec = self._preprocess(wav)

        if self.augment is not None:
            spec = self.augment(spec)

        return spec, self._labels[idx]

    # ------------------------------------------------------------------
    # Convenience attributes
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Number of target classes (35 for GSC v2)."""
        return len(GSC_CLASSES)

    @property
    def class_names(self) -> List[str]:
        """Ordered list of class name strings."""
        return GSC_CLASSES
