"""Raw PSG -> model-ready arrays (Fig. 1 of the paper).

Pipeline for one recording
--------------------------
1. read the EDF (``pyedflib``) and the 30-s hypnogram,
2. per-subject z-score of every channel over the whole recording,
3. down-sample 512 Hz -> 64 Hz (linear interpolation, as in the paper; ``method="decimate"``
   adds an anti-aliasing filter and is recommended for new experiments),
4. PPG peak detection -> inter-beat intervals -> HRV tachogram resampled to 64 Hz,
5. segmentation into 30-s epochs aligned with the hypnogram, stacking the channels
   ``["PPG", "RF", "RC", "RA", "HRV"]`` -> ``(n_epochs, 1920, 5)``.

The Natus channel labels differ between sites, so they are passed explicitly
(see ``scripts/preprocess_edf.py --help``). Hypnograms are read from a CSV with
either an ``epoch`` or an ``onset`` (seconds) column and a ``stage`` column that
uses AASM labels (``W, N1, N2, N3, R`` or ``0..4``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import decimate, find_peaks

from . import CHANNEL_NAMES, EPOCH_SECONDS

log = logging.getLogger(__name__)

STAGE_MAP = {
    "W": 0, "WAKE": 0, "0": 0, "SLEEP-S0": 0,
    "N1": 1, "S1": 1, "1": 1, "SLEEP-S1": 1,
    "N2": 2, "S2": 2, "2": 2, "SLEEP-S2": 2,
    "N3": 3, "S3": 3, "S4": 3, "3": 3, "4": 3, "SLEEP-S3": 3, "SLEEP-S4": 3,
    "R": 4, "REM": 4, "5": 4, "SLEEP-REM": 4,
}


@dataclass
class ChannelMap:
    """EDF labels of the four raw channels (HRV is derived from PPG)."""

    ppg: str = "Pleth"
    flow: str = "Flow"
    chest: str = "Chest"
    abdomen: str = "Abdomen"

    def as_dict(self) -> Dict[str, str]:
        return {"PPG": self.ppg, "RF": self.flow, "RC": self.chest, "RA": self.abdomen}


@dataclass
class PreprocessConfig:
    target_fs: int = 64
    epoch_seconds: int = EPOCH_SECONDS
    downsample_method: str = "linear"          # "linear" (paper) | "decimate" (anti-aliased)
    ppg_min_rr_s: float = 0.33                 # >= 180 bpm
    ppg_max_rr_s: float = 2.0                  # <= 30 bpm
    drop_stages: Tuple[str, ...] = ("?", "UNKNOWN", "MOVEMENT", "ARTIFACT", "L", "N/A")
    channel_map: ChannelMap = field(default_factory=ChannelMap)


# ---------------------------------------------------------------------------
# EDF / hypnogram readers
# ---------------------------------------------------------------------------
def read_edf_channels(path: str | Path, labels: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Return ``{label: signal}`` and ``{label: fs}`` for the requested EDF channels."""
    import pyedflib  # optional dependency

    sig: Dict[str, np.ndarray] = {}
    fs: Dict[str, float] = {}
    with pyedflib.EdfReader(str(path)) as f:
        edf_labels = f.getSignalLabels()
        for lab in labels:
            if lab not in edf_labels:
                raise KeyError(f"channel '{lab}' not in {path}; available: {edf_labels}")
            i = edf_labels.index(lab)
            sig[lab] = f.readSignal(i).astype(np.float64)
            fs[lab] = float(f.getSampleFrequency(i))
    return sig, fs


def read_hypnogram(path: str | Path, epoch_seconds: int = EPOCH_SECONDS) -> np.ndarray:
    """Read a hypnogram CSV -> int array of per-epoch stages (-1 = unscored)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    stage_col = cols.get("stage") or cols.get("label") or cols.get("sleep_stage")
    if stage_col is None:
        raise ValueError(f"{path}: need a 'stage' column")
    stages = df[stage_col].astype(str).str.strip().str.upper()
    codes = np.array([STAGE_MAP.get(s, -1) for s in stages], dtype=np.int64)
    if "epoch" in cols:
        idx = df[cols["epoch"]].astype(int).to_numpy()
    elif "onset" in cols:
        idx = np.round(df[cols["onset"]].astype(float).to_numpy() / epoch_seconds).astype(int)
    else:
        idx = np.arange(len(df))
    idx = idx - idx.min()
    hyp = -np.ones(idx.max() + 1, dtype=np.int64)
    hyp[idx] = codes
    return hyp


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    return (x - x.mean()) / max(x.std(), eps)


def downsample(x: np.ndarray, fs_in: float, fs_out: float, method: str = "linear") -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return x
    n_out = int(round(len(x) * fs_out / fs_in))
    if method == "decimate":
        q = fs_in / fs_out
        if abs(q - round(q)) < 1e-9 and q > 1:
            return decimate(x, int(round(q)), ftype="fir", zero_phase=True)[:n_out]
    # linear interpolation (Parker et al., 1983) - used for the published results
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x)


def ppg_peaks(ppg: np.ndarray, fs: float, min_rr_s: float = 0.33) -> np.ndarray:
    """Systolic peak indices (scipy; use neurokit2 for a more robust detector if installed)."""
    try:
        import neurokit2 as nk

        clean = nk.ppg_clean(ppg, sampling_rate=int(fs))
        info = nk.ppg_findpeaks(clean, sampling_rate=int(fs))
        return np.asarray(info["PPG_Peaks"], dtype=int)
    except Exception:  # pragma: no cover - fallback path
        x = ppg - np.median(ppg)
        peaks, _ = find_peaks(x, distance=int(min_rr_s * fs), prominence=0.3 * np.std(x))
        return peaks


def hrv_tachogram(ppg: np.ndarray, fs: float, cfg: PreprocessConfig) -> np.ndarray:
    """Instantaneous inter-beat interval series interpolated onto the PPG time grid (seconds)."""
    pk = ppg_peaks(ppg, fs, cfg.ppg_min_rr_s)
    if len(pk) < 3:
        return np.zeros_like(ppg)
    rr = np.diff(pk) / fs
    t_rr = pk[1:] / fs
    ok = (rr >= cfg.ppg_min_rr_s) & (rr <= cfg.ppg_max_rr_s)
    if ok.sum() < 3:
        return np.zeros_like(ppg)
    t = np.arange(len(ppg)) / fs
    return np.interp(t, t_rr[ok], rr[ok])


# ---------------------------------------------------------------------------
# Recording -> epochs
# ---------------------------------------------------------------------------
def process_recording(edf_path: str | Path, hypnogram_path: str | Path, cfg: PreprocessConfig,
                      subject_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y, subject)`` for one recording: ``x`` is ``(n, T, 5)`` float32."""
    cmap = cfg.channel_map.as_dict()
    sig, fs = read_edf_channels(edf_path, list(cmap.values()))
    hyp = read_hypnogram(hypnogram_path, cfg.epoch_seconds)

    T = cfg.target_fs * cfg.epoch_seconds
    chans: Dict[str, np.ndarray] = {}
    for name, lab in cmap.items():
        x = downsample(sig[lab], fs[lab], cfg.target_fs, cfg.downsample_method)
        chans[name] = zscore(x)
    ppg_raw = downsample(sig[cmap["PPG"]], fs[cmap["PPG"]], cfg.target_fs, cfg.downsample_method)
    chans["HRV"] = zscore(hrv_tachogram(ppg_raw, cfg.target_fs, cfg))

    n_epochs = min(len(hyp), min(len(v) for v in chans.values()) // T)
    x = np.stack([chans[c][: n_epochs * T].reshape(n_epochs, T) for c in CHANNEL_NAMES], axis=-1)
    y = hyp[:n_epochs]
    keep = y >= 0
    x, y = x[keep].astype(np.float32), y[keep]
    subject = np.array([subject_id] * len(y))
    log.info("%s: %d scored epochs (%s)", subject_id, len(y), np.bincount(y, minlength=5))
    return x, y, subject


def split_subjects(subjects: Sequence[str], fractions=(0.70, 0.15, 0.15), seed: int = 0) -> Dict[str, List[str]]:
    """Inter-patient split of subject ids into train / val / test."""
    rng = np.random.default_rng(seed)
    s = np.array(sorted(set(subjects)))
    rng.shuffle(s)
    n = len(s)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    return {"train": s[:n_tr].tolist(), "val": s[n_tr:n_tr + n_va].tolist(), "test": s[n_tr + n_va:].tolist()}


def save_splits(out_dir: str | Path, x: np.ndarray, y: np.ndarray, subject: np.ndarray,
                split_map: Dict[str, List[str]], disorder: Optional[np.ndarray] = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, subs in split_map.items():
        m = np.isin(subject, subs)
        np.save(out_dir / f"{split}_data.npy", x[m])
        np.save(out_dir / f"{split}_label.npy", y[m])
        np.save(out_dir / f"{split}_subject.npy", subject[m])
        if disorder is not None:
            np.save(out_dir / f"{split}_disorder.npy", disorder[m])
        log.info("%s: %d epochs from %d subjects -> %s", split, m.sum(), len(subs), out_dir)
