"""Data loading and preparation.

All functions work on NumPy arrays shaped ``(n_epochs, n_samples, n_channels)`` with
the fixed channel order :data:`vit_sleep.CHANNEL_NAMES` = ``["PPG", "RF", "RC", "RA", "HRV"]``.

The heavy lifting (segmentation of raw PSG into 30-s epochs, HRV extraction,
per-subject z-scoring and 512 -> 64 Hz down-sampling) is done once by
``scripts/preprocess_edf.py``; here we only resample to a fixed length, apply the
per-epoch normalisation used at training time and build ``tf.data`` pipelines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample

from . import CHANNEL_NAMES, CLASS_NAMES, NUM_CLASSES

log = logging.getLogger(__name__)

# Historical export names (as used by the original notebooks) -> canonical names.
# Both spellings are accepted, and files are searched in ``data_dir`` *and* in its parent
# (the repository root, where the original notebooks kept them).
LEGACY_FILENAMES = {
    "train_data.npy": ["train_data_5calss.npy", "train_data_5class.npy"],
    "train_label.npy": ["train_label_5calss.npy", "train_label_5class.npy"],
    "val_data.npy": ["val_data_5calss.npy", "val_data_5class.npy"],
    "val_label.npy": ["val_label_5calss.npy", "val_label_5class.npy"],
    "test_data.npy": ["test_data_5calss.npy", "test_data_5class.npy"],
    "test_label.npy": ["test_label_5calss.npy", "test_label_5class.npy"],
}
DISORDERS = ("OSA", "Insomnia", "Hypersomnia")   # legacy per-disorder test files: test_data_5calss_<disorder>.npy


@dataclass
class SplitData:
    """One data split (train / val / test).

    ``subject``      true subject ids (None when unknown) - used for subject-level bootstrap CIs.
    ``sequence_id``  grouping of epochs into *recordings in chronological order*; equals ``subject``
                     when subject ids exist, otherwise a *pseudo* recording id inferred from the
                     stored order (see :func:`infer_pseudo_sequences`). Used by everything that
                     needs the hypnogram (transition matrix, HMM smoothing, context windows).
    """

    x: np.ndarray                       # (N, T, C) float32
    y: np.ndarray                       # (N,) int
    subject: Optional[np.ndarray] = None  # (N,) subject id (chronological order within subject!)
    disorder: Optional[np.ndarray] = None  # (N,) str, e.g. "OSA"
    channels: Tuple[str, ...] = tuple(CHANNEL_NAMES)
    sequence_id: Optional[np.ndarray] = None  # (N,) pseudo-recording id when subject is None
    sequence_kind: str = "none"          # "subject" | "pseudo" | "none"

    def __len__(self) -> int:
        return len(self.y)

    @property
    def seq(self) -> Optional[np.ndarray]:
        """Recording grouping usable for sequence analyses (true subjects if known, else pseudo ids)."""
        return self.subject if self.subject is not None else self.sequence_id

    def subset(self, idx: np.ndarray) -> "SplitData":
        return SplitData(
            x=self.x[idx] if self.x.shape[0] else self.x,      # labels-only objects keep their empty x
            y=self.y[idx],
            subject=None if self.subject is None else self.subject[idx],
            disorder=None if self.disorder is None else self.disorder[idx],
            channels=self.channels,
            sequence_id=None if self.sequence_id is None else self.sequence_id[idx],
            sequence_kind=self.sequence_kind,
        )

    def select_channels(self, channels: Sequence[str]) -> "SplitData":
        idx = channel_indices(channels, self.channels)
        return SplitData(self.x[..., idx], self.y, self.subject, self.disorder, tuple(channels),
                         self.sequence_id, self.sequence_kind)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_WARNED_NO_SUBJECT: set = set()


def _find(data_dir: Path, name: str, extra_candidates: Sequence[str] = ()) -> Optional[Path]:
    """Locate ``name`` (or one of its legacy spellings) in ``data_dir`` or in its parent directory."""
    candidates = [name] + LEGACY_FILENAMES.get(name, []) + list(extra_candidates)
    for folder in (data_dir, data_dir.parent):
        for c in candidates:
            p = folder / c
            if p.exists():
                return p
    return None


def resolve_data_dir(root: str | Path) -> Path:
    """Return ``<root>/data`` if it holds the arrays, else ``<root>`` when the legacy files live there."""
    root = Path(root)
    if _find(root / "data", "train_data.npy") is not None:
        return root / "data"
    return root


def available_disorders(data_dir: str | Path, split: str = "test") -> List[str]:
    """Disorders for which legacy per-disorder files (``test_data_5calss_OSA.npy`` ...) exist."""
    data_dir = Path(data_dir)
    return [d for d in DISORDERS
            if _find(data_dir, f"{split}_data_5calss_{d}.npy", [f"{split}_data_5class_{d}.npy", f"{split}_data_{d}.npy"])]


def load_split(data_dir: str | Path, split: str, disorder: Optional[str] = None) -> SplitData:
    """Load ``{split}_data.npy`` / ``{split}_label.npy`` (+ optional subject / disorder arrays).

    Legacy names (``train_data_5calss.npy`` ...) are accepted, in ``data_dir`` or its parent.
    ``disorder`` selects the legacy per-disorder files (``test_data_5calss_OSA.npy`` ...) when a
    ``{split}_disorder.npy`` array is not available.
    """
    data_dir = Path(data_dir)
    if disorder is not None and _find(data_dir, f"{split}_disorder.npy") is None:
        x_path = _find(data_dir, f"{split}_data_5calss_{disorder}.npy",
                       [f"{split}_data_5class_{disorder}.npy", f"{split}_data_{disorder}.npy"])
        y_path = _find(data_dir, f"{split}_label_5calss_{disorder}.npy",
                       [f"{split}_label_5class_{disorder}.npy", f"{split}_label_{disorder}.npy"])
    else:
        x_path = _find(data_dir, f"{split}_data.npy")
        y_path = _find(data_dir, f"{split}_label.npy")
    if x_path is None or y_path is None:
        raise FileNotFoundError(
            f"Could not find {split}{'/' + disorder if disorder else ''} arrays in {data_dir} or {data_dir.parent} "
            f"(accepted names: {split}_data.npy, {split}_data_5calss.npy; see data/README.md)")

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path).astype(np.int64).ravel()
    subj_path = _find(data_dir, f"{split}_subject.npy")
    dis_path = _find(data_dir, f"{split}_disorder.npy")
    subject = np.load(subj_path, allow_pickle=True).ravel() if subj_path else None
    dis = np.load(dis_path, allow_pickle=True).ravel() if dis_path else None
    if subject is None and disorder is None and split not in _WARNED_NO_SUBJECT:
        _WARNED_NO_SUBJECT.add(split)
        log.warning("%s split has no subject ids (%s); pseudo-recordings will be inferred from the stored "
                    "epoch order when it is chronological - see load_experiment_data(pseudo_sequences=...).",
                    split, y_path.name)
    sd = SplitData(np.asarray(x), y, subject, dis, sequence_kind="subject" if subject is not None else "none")
    if disorder is not None and dis is not None:
        sd = sd.subset(np.where(dis == disorder)[0])
    elif disorder is not None:
        sd.disorder = np.array([disorder] * len(y))
    return sd


def _load_consistent_alternative(data_dir: Path, split: str, ref_y: np.ndarray) -> Optional[Tuple[SplitData, str]]:
    """Among the alternative label files of ``split`` (``{split}_label_5calss.npy`` ...), return the first whose
    coding is consistent with ``ref_y`` together with its *matching* data file (same suffix)."""
    for suffix in ("_5calss", "_5class", ""):
        y_path, x_path = None, None
        for folder in (data_dir, data_dir.parent):
            yp, xp = folder / f"{split}_label{suffix}.npy", folder / f"{split}_data{suffix}.npy"
            if yp.exists() and xp.exists():
                y_path, x_path = yp, xp
                break
        if y_path is None:
            continue
        y = np.load(y_path).astype(np.int64).ravel()
        if check_label_consistency(ref_y, y)["ok"]:
            x = np.load(x_path, mmap_mode="r")
            if x.shape[0] != len(y):
                continue
            return SplitData(np.asarray(x), y, None, None, sequence_kind="none"), f"{x_path.name} + {y_path.name}"
    return None


def load_disorder_test_sets(data_dir: str | Path, channels: Sequence[str], target_len: int = 2000,
                            normalisation: str = "minmax", resample_method: str = "fft",
                            split: str = "test") -> Dict[str, SplitData]:
    """Per-disorder test sets (Table 2 of the paper) from the legacy ``test_data_5calss_<disorder>.npy`` files
    or, when ``test_disorder.npy`` exists, by filtering the main test split."""
    data_dir = Path(data_dir)
    out: Dict[str, SplitData] = {}
    if _find(data_dir, f"{split}_disorder.npy") is not None:
        full = load_split(data_dir, split)
        names = [str(d) for d in np.unique(full.disorder)]
    else:
        names = available_disorders(data_dir, split)
        full = None
    for d in names:
        sd = load_split(data_dir, split, disorder=d) if full is None else full.subset(np.where(full.disorder == d)[0])
        x = preprocess_epochs(sd.x, target_len, normalisation, resample_method)
        out[d] = SplitData(x, sd.y, sd.subject, sd.disorder, sd.channels).select_channels(channels)
    return out


# ---------------------------------------------------------------------------
# Pre-processing (resampling + per-epoch normalisation)
# ---------------------------------------------------------------------------
def resample_epochs(x: np.ndarray, target_len: int, method: str = "fft", chunk: int = 2048) -> np.ndarray:
    """Resample every epoch to ``target_len`` samples along the time axis.

    method="fft"      scipy.signal.resample (used for the published results, 1921 -> 2000)
    method="truncate" crop/zero-pad (recommended when 1921 -> 1920)
    method="linear"   linear interpolation
    """
    x = np.asarray(x)
    n, t, c = x.shape
    if t == target_len:
        return np.array(x, dtype=np.float32)        # copy: input may be a read-only memmap
    out = np.empty((n, target_len, c), dtype=np.float32)
    if method == "truncate":
        m = min(t, target_len)
        out[:] = 0
        out[:, :m] = x[:, :m]
        return out
    if method == "linear":
        src = np.linspace(0, 1, t)
        dst = np.linspace(0, 1, target_len)
        for i in range(n):
            for k in range(c):
                out[i, :, k] = np.interp(dst, src, x[i, :, k])
        return out
    for s in range(0, n, chunk):
        out[s:s + chunk] = resample(np.asarray(x[s:s + chunk], dtype=np.float32), target_len, axis=1)
    return out


def normalise_epochs(x: np.ndarray, method: str = "minmax", eps: float = 1e-8) -> np.ndarray:
    """Per-epoch, per-channel normalisation (applied after the per-subject z-scoring of preprocessing)."""
    x = np.asarray(x, dtype=np.float32)
    if method == "none":
        return x
    if method == "minmax":
        lo = x.min(axis=1, keepdims=True)
        hi = x.max(axis=1, keepdims=True)
        return (x - lo) / np.maximum(hi - lo, eps)
    if method == "zscore":
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        return (x - mu) / np.maximum(sd, eps)
    raise ValueError(f"unknown normalisation '{method}'")


def preprocess_epochs(
    x: np.ndarray, target_len: int = 2000, normalisation: str = "minmax", resample_method: str = "fft"
) -> np.ndarray:
    x = resample_epochs(x, target_len, method=resample_method)
    x = np.nan_to_num(x, copy=False)
    return normalise_epochs(x, normalisation)


def channel_indices(channels: Sequence[str], available: Sequence[str] = CHANNEL_NAMES) -> List[int]:
    missing = [c for c in channels if c not in available]
    if missing:
        raise KeyError(f"unknown channel(s) {missing}; available: {list(available)}")
    return [list(available).index(c) for c in channels]


# ---------------------------------------------------------------------------
# Label sanity checks and pseudo-recordings (data sets exported without subject ids)
# ---------------------------------------------------------------------------
def inspect_label_files(data_dir: str | Path) -> "pd.DataFrame":
    """Class counts of every ``*label*.npy`` in ``data_dir`` and its parent - to spot encoding mismatches."""
    import pandas as pd

    data_dir = Path(data_dir)
    rows = []
    seen = set()
    for folder in (data_dir, data_dir.parent):
        for p in sorted(folder.glob("*label*.npy")):
            if p.name in seen:
                continue
            seen.add(p.name)
            y = np.load(p, mmap_mode="r")
            vals, cnt = np.unique(np.asarray(y).ravel(), return_counts=True)
            row = {"file": p.name, "n": int(np.asarray(y).size), "dtype": str(np.asarray(y).dtype),
                   "classes": vals.tolist()}
            for v, c in zip(vals, cnt):
                row[f"pct_{int(v)}"] = round(100 * c / cnt.sum(), 1)
            rows.append(row)
    return pd.DataFrame(rows).set_index("file")


def check_label_consistency(train_y: np.ndarray, test_y: np.ndarray, max_shift: float = 0.15) -> Dict[str, object]:
    """Compare class sets and class proportions of two splits.

    Returns ``{"ok": bool, "message": str, ...}``. A different set of class codes, or a class whose
    share differs by more than ``max_shift`` (absolute), indicates that the label files use different
    encodings (e.g. N1/N2 merged) and that every metric computed on them would be meaningless.
    """
    tr = np.bincount(np.asarray(train_y).astype(int), minlength=NUM_CLASSES) / len(train_y)
    te = np.bincount(np.asarray(test_y).astype(int), minlength=NUM_CLASSES) / len(test_y)
    tr_classes, te_classes = set(np.unique(train_y).tolist()), set(np.unique(test_y).tolist())
    shift = np.abs(tr[:NUM_CLASSES] - te[:NUM_CLASSES])
    problems = []
    if tr_classes != te_classes:
        problems.append(f"class codes differ: train {sorted(tr_classes)} vs test {sorted(te_classes)}")
    if shift.max() > max_shift:
        k = int(shift.argmax())
        problems.append(f"share of class {k} ({CLASS_NAMES[k] if k < NUM_CLASSES else k}) is {100*tr[k]:.1f}% in train "
                        f"but {100*te[k]:.1f}% in test")
    msg = ("labels consistent" if not problems else
           "LABEL ENCODING MISMATCH - " + "; ".join(problems) +
           ". Metrics on this test set are meaningless until train and test use the same 0..4 coding "
           "(0 Wake, 1 N1, 2 N2, 3 N3, 4 REM). Run vit_sleep.data.inspect_label_files(DATA_DIR) to see every "
           "label file and delete/rename the wrong one so the loader falls back to the correct file.")
    return {"ok": not problems, "message": msg, "train_pct": np.round(100 * tr, 1).tolist(),
            "test_pct": np.round(100 * te, 1).tolist()}


def chronology_score(labels: np.ndarray) -> float:
    """Fraction of consecutive epochs with the same stage.

    Real hypnograms give ~0.8-0.9 (stages persist for minutes); a shuffled sequence gives
    ``sum_k p_k^2`` (~0.3 for this class distribution). Used to decide whether the stored order
    of an export without subject ids can be treated as chronological.
    """
    y = np.asarray(labels)
    return float((y[1:] == y[:-1]).mean()) if len(y) > 1 else 0.0


def shuffled_chronology_baseline(labels: np.ndarray) -> float:
    p = np.bincount(np.asarray(labels).astype(int)) / len(labels)
    return float((p ** 2).sum())


def locate_subsequence(full: np.ndarray, part: np.ndarray) -> Optional[int]:
    """Start index of ``part`` as a contiguous slice of ``full`` (None if absent). Both 1-D int arrays."""
    full, part = np.asarray(full), np.asarray(part)
    m = len(part)
    if m == 0 or m > len(full):
        return None
    # candidates: positions matching the first 8 elements, then verify fully
    k = min(8, m)
    cand = np.where(full[: len(full) - m + 1] == part[0])[0]
    for i in cand:
        if np.array_equal(full[i:i + k], part[:k]) and np.array_equal(full[i:i + m], part):
            return int(i)
    return None


def infer_pseudo_sequences(data_dir: str | Path, split: str, y: np.ndarray,
                           min_chronology: float = 0.6) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Infer pseudo-recording ids (and disorder labels) for a split exported without subject ids.

    1. The stored order is accepted as chronological if :func:`chronology_score` >= ``min_chronology``
       (a shuffled export is rejected -> returns ``(None, None, info)``).
    2. If legacy per-disorder files exist (``{split}_label_5calss_OSA.npy`` ...) and each disorder's
       label sequence is found as a contiguous slice of the full label array, the slice boundaries
       become pseudo-recording boundaries and a per-epoch ``disorder`` array is returned.
       Otherwise the whole split is one pseudo-recording.

    Caveat (state it in the paper): subject boundaries *inside* a pseudo-recording are unknown, so
    transition statistics contain at most (n_subjects - 1) spurious transitions per split and context
    windows may straddle two subjects near those boundaries.
    """
    data_dir = Path(data_dir)
    y = np.asarray(y).astype(int)
    score, base = chronology_score(y), shuffled_chronology_baseline(y)
    info = {"chronology_score": score, "shuffled_baseline": base, "accepted": score >= min_chronology,
            "disorder_slices": {}}
    if not info["accepted"]:
        info["message"] = (f"{split}: stored order is NOT chronological (self-transition rate {score:.2f}, "
                           f"shuffled baseline {base:.2f}); no pseudo-recordings.")
        return None, None, info

    seq = np.zeros(len(y), dtype=np.int64)
    disorder = None
    found = {}
    for d in DISORDERS:
        p = _find(data_dir, f"{split}_label_5calss_{d}.npy", [f"{split}_label_5class_{d}.npy", f"{split}_label_{d}.npy"])
        if p is None:
            continue
        part = np.load(p).astype(int).ravel()
        start = locate_subsequence(y, part)
        if start is not None:
            found[d] = (start, start + len(part))
    if found and sum(e - s for s, e in found.values()) <= len(y):
        disorder = np.array(["unknown"] * len(y), dtype=object)
        for k, (d, (s, e)) in enumerate(sorted(found.items(), key=lambda kv: kv[1][0])):
            seq[s:e] = k + 1
            disorder[s:e] = d
        info["disorder_slices"] = found
        info["message"] = (f"{split}: order chronological (self-transition rate {score:.2f} vs {base:.2f} shuffled); "
                           f"{len(found)} pseudo-recordings from disorder files {sorted(found)}"
                           + ("" if (disorder != "unknown").all() else f"; {(disorder == 'unknown').sum()} epochs not in any disorder file"))
        disorder = disorder.astype(str)
    else:
        info["message"] = (f"{split}: order chronological (self-transition rate {score:.2f} vs {base:.2f} shuffled); "
                           f"whole split treated as one pseudo-recording (disorder files not found / not contiguous)")
    return seq, disorder, info


def block_split(n: int, val_fraction: float = 0.15, n_blocks: int = 20, seed: int = 0,
                groups: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Contiguous-block train/validation split (proxy for a subject-wise split when only the order is known).

    Each recording (``groups``) is cut into ``n_blocks`` contiguous blocks and ~``val_fraction`` of the
    blocks go to validation, so validation epochs are (mostly) not immediate neighbours of training epochs.
    """
    rng = np.random.default_rng(seed)
    groups = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
    is_val = np.zeros(n, dtype=bool)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        nb = max(2, min(n_blocks, len(idx) // 50))
        blocks = np.array_split(idx, nb)
        n_val = max(1, int(round(val_fraction * nb)))
        for b in rng.choice(nb, n_val, replace=False):
            is_val[blocks[b]] = True
    return np.where(~is_val)[0], np.where(is_val)[0]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def subject_split(
    subject: np.ndarray, val_fraction: float = 0.15, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Subject-wise (inter-patient) train/validation index split."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject)
    rng.shuffle(subjects)
    n_val = max(1, int(round(val_fraction * len(subjects))))
    val_subjects = set(subjects[:n_val].tolist())
    is_val = np.array([s in val_subjects for s in subject])
    return np.where(~is_val)[0], np.where(is_val)[0]


def load_experiment_data(
    data_dir: str | Path,
    channels: Sequence[str],
    target_len: int = 2000,
    normalisation: str = "minmax",
    val_fraction: float = 0.15,
    seed: int = 0,
    resample_method: str = "fft",
    pseudo_sequences: str | bool = "auto",
    val_split: str = "random",
    check_labels: bool = True,
) -> Dict[str, SplitData]:
    """Load, preprocess and channel-select train / val / test splits.

    * The train/test split on disk is used as is (it was made subject-wise when the arrays were exported).
    * If ``val_*.npy`` is absent, a validation set is carved out of *train*: subject-wise when subject ids
      exist; otherwise ``val_split="random"`` (epoch level, reproducible, the setting used for the reference
      runs) or ``val_split="blocks"`` (contiguous blocks - a better proxy of a subject split when the stored
      order is chronological, but the validation set then differs from earlier runs).
    * ``pseudo_sequences``: when subject ids are missing, ``"auto"``/``True`` infers pseudo-recording ids and
      disorder labels from the stored order and the per-disorder files (:func:`infer_pseudo_sequences`);
      ``False`` disables it. The result is exposed as ``SplitData.sequence_id`` / ``.seq`` and enables the
      hypnogram-based analyses at segment level.
    * ``check_labels``: raises ``ValueError`` when train and test label codings are inconsistent.
    """
    data_dir = Path(data_dir)
    out: Dict[str, SplitData] = {}
    train = load_split(data_dir, "train")
    test = load_split(data_dir, "test")
    if check_labels:
        chk = check_label_consistency(train.y, test.y)
        if not chk["ok"]:
            # try the alternative export names (test_label_5calss.npy + test_data_5calss.npy, ...) before giving up
            alt = _load_consistent_alternative(data_dir, "test", train.y)
            if alt is None:
                raise ValueError(chk["message"])
            test, used = alt
            log.warning("PRIMARY TEST LABEL FILE HAS A DIFFERENT CODING (%s). Using the consistent pair %s instead - "
                        "move the wrong file out of the data folder to silence this.", chk["message"].split(".")[0], used)

    # pseudo-recordings (segment-level data sets) -------------------------------------------
    if pseudo_sequences and train.subject is None:
        for split_name, split in (("train", train), ("test", test)):
            seq, dis, info = infer_pseudo_sequences(data_dir, split_name, split.y)
            log.info(info["message"])
            if seq is not None:
                split.sequence_id, split.sequence_kind = seq, "pseudo"
                if split.disorder is None and dis is not None:
                    split.disorder = dis
            elif pseudo_sequences is True:
                raise ValueError(info["message"])

    # labels-only copy of the complete training hypnograms (before any validation carve-out) for
    # transition statistics - a random validation subset would otherwise punch holes into the sequences
    out["train_hypnogram"] = SplitData(np.zeros((0, target_len, len(train.channels)), np.float32), train.y.copy(),
                                       train.subject, train.disorder, train.channels, train.sequence_id,
                                       train.sequence_kind).select_channels(channels)
    try:
        val = load_split(data_dir, "val")
    except FileNotFoundError:
        if train.subject is not None:
            tr_idx, va_idx = subject_split(train.subject, val_fraction, seed)
        elif val_split == "blocks":
            log.warning("No subject ids: validation = contiguous blocks of the training sequence (val_split='blocks').")
            tr_idx, va_idx = block_split(len(train), val_fraction, seed=seed, groups=train.sequence_id)
        else:
            log.warning("No subject ids: validation split is a random epoch subset of train (NOT inter-patient; "
                        "use val_split='blocks' or provide val_*.npy for a stricter split).")
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(train))
            n_val = int(val_fraction * len(train))
            va_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        val = train.subset(va_idx)
        train = train.subset(tr_idx)
        if val.sequence_kind == "pseudo" and val_split != "blocks":
            val.sequence_id, val.sequence_kind = None, "none"     # a random subset has no chronology
            train.sequence_kind = "pseudo-gapped"                 # 15% of epochs missing -> use train_hypnogram

    for name, split in (("train", train), ("val", val), ("test", test)):
        x = preprocess_epochs(split.x, target_len, normalisation, resample_method)
        sd = SplitData(x, split.y, split.subject, split.disorder, split.channels,
                       split.sequence_id, split.sequence_kind).select_channels(channels)
        out[name] = sd
        log.info("%s: %s epochs, sequences=%s (%s)", name, x.shape,
                 "n/a" if sd.seq is None else len(np.unique(sd.seq)), sd.sequence_kind)
    return out


# ---------------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------------
def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = False,
                 seed: int = 0, one_hot: bool = True, sample_weight: Optional[np.ndarray] = None):
    import tensorflow as tf

    labels = tf.keras.utils.to_categorical(y, NUM_CLASSES).astype(np.float32) if one_hot else y
    tensors = (x.astype(np.float32), labels) if sample_weight is None else (x.astype(np.float32), labels, sample_weight.astype(np.float32))
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if shuffle:
        ds = ds.shuffle(min(len(x), 20_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(y: np.ndarray, mode: Optional[str] = "balanced") -> Optional[Dict[int, float]]:
    """Per-class weights. ``balanced`` = N / (K * n_k); ``sqrt`` = sqrt of that (milder)."""
    if mode is None:
        return None
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    w = len(y) / (NUM_CLASSES * np.maximum(counts, 1))
    if mode == "sqrt":
        w = np.sqrt(w)
    elif mode != "balanced":
        raise ValueError(mode)
    return {k: float(v) for k, v in enumerate(w)}


# ---------------------------------------------------------------------------
# Sequences of consecutive epochs (temporal context)
# ---------------------------------------------------------------------------
def subject_runs(subject: np.ndarray) -> List[np.ndarray]:
    """Indices of each subject's epochs, in array order (assumed chronological)."""
    runs = []
    for s in np.unique(subject):
        runs.append(np.where(subject == s)[0])
    return runs


def make_context_windows(subject: np.ndarray, context: int = 10) -> np.ndarray:
    """Index matrix ``(N, 2*context+1)`` of the neighbouring epochs of every epoch.

    Windows never cross subject boundaries; positions outside the recording are
    filled with ``-1`` (to be masked). The centre column is the epoch itself.
    """
    n = len(subject)
    L = 2 * context + 1
    win = -np.ones((n, L), dtype=np.int64)
    for run in subject_runs(subject):
        m = len(run)
        for j, k in enumerate(range(-context, context + 1)):
            src = np.arange(m) + k
            ok = (src >= 0) & (src < m)
            win[run[ok], j] = run[src[ok]]
    return win


def gather_windows(x: np.ndarray, windows: np.ndarray, pad_value: float = 0.0):
    """Gather ``x[windows]`` with ``-1`` -> ``pad_value``. Returns (data, mask)."""
    mask = windows >= 0
    safe = np.where(mask, windows, 0)
    out = x[safe]
    out[~mask] = pad_value
    return out, mask
