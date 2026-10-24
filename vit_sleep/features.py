"""Hand-crafted autonomic features per 30-s epoch - used for the N1 error analysis (Reviewer comment 3).

The question we answer with these features: *is the cardiorespiratory signature of N1
distinguishable from Wake and N2 within a single 30-s epoch at all?* If simple,
physiologically grounded descriptors (heart rate, HRV, breathing rate/variability,
thoraco-abdominal coupling) show near-chance separability for N1 vs its neighbours,
the model's N1 "collapse" reflects an information limit of isolated epochs rather
than a representational failure of the ViT.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score

from . import CLASS_NAMES, EPOCH_SECONDS


def _peaks(sig: np.ndarray, fs: float, min_dist_s: float, prominence_frac: float = 0.25) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.median(sig)
    sd = sig.std()
    if sd < 1e-9:
        return np.array([], dtype=int)
    pk, _ = find_peaks(sig, distance=max(1, int(min_dist_s * fs)), prominence=prominence_frac * sd)
    return pk


def _cv(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return float(v.std() / abs(v.mean())) if len(v) > 1 and abs(v.mean()) > 1e-9 else np.nan


def ppg_features(ppg: np.ndarray, fs: float) -> Dict[str, float]:
    pk = _peaks(ppg, fs, min_dist_s=0.33)
    f = {"hr_mean": np.nan, "hr_std": np.nan, "sdnn": np.nan, "rmssd": np.nan, "ppg_amp_cv": np.nan, "n_beats": len(pk)}
    if len(pk) < 4:
        return f
    ibi = np.diff(pk) / fs
    ok = (ibi > 0.33) & (ibi < 2.0)
    ibi = ibi[ok]
    if len(ibi) < 3:
        return f
    f["hr_mean"] = float(60.0 / ibi.mean())
    f["hr_std"] = float((60.0 / ibi).std())
    f["sdnn"] = float(ibi.std() * 1000)
    f["rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000)
    troughs = [np.min(ppg[pk[i - 1]:pk[i]]) for i in range(1, len(pk))]
    amp = ppg[pk[1:]] - np.asarray(troughs)
    f["ppg_amp_cv"] = _cv(amp)
    return f


def resp_features(resp: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    pk = _peaks(resp, fs, min_dist_s=1.5)
    f = {f"{prefix}_rate": np.nan, f"{prefix}_bb_cv": np.nan, f"{prefix}_amp_cv": np.nan}
    if len(pk) < 3:
        return f
    bb = np.diff(pk) / fs
    f[f"{prefix}_rate"] = float(60.0 / bb.mean())
    f[f"{prefix}_bb_cv"] = _cv(bb)
    troughs = [np.min(resp[pk[i - 1]:pk[i]]) for i in range(1, len(pk))]
    f[f"{prefix}_amp_cv"] = _cv(resp[pk[1:]] - np.asarray(troughs))
    return f


def coupling_features(rc: np.ndarray, ra: np.ndarray) -> Dict[str, float]:
    """Thoraco-abdominal coupling: correlation and lag of max cross-correlation (paradoxical breathing in OSA)."""
    rc = (rc - rc.mean()) / (rc.std() + 1e-9)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    r = float(np.corrcoef(rc, ra)[0, 1]) if len(rc) > 2 else np.nan
    return {"rc_ra_corr": r}


def epoch_features(x: np.ndarray, channels: Sequence[str], epoch_seconds: int = EPOCH_SECONDS) -> pd.DataFrame:
    """Feature table for epochs ``x`` of shape ``(N, T, C)`` with channel names ``channels``."""
    fs = x.shape[1] / epoch_seconds
    ch = {c: i for i, c in enumerate(channels)}
    rows: List[Dict[str, float]] = []
    for i in range(len(x)):
        f: Dict[str, float] = {}
        if "PPG" in ch:
            f.update(ppg_features(x[i, :, ch["PPG"]], fs))
        for name in ("RC", "RA", "RF"):
            if name in ch:
                f.update(resp_features(x[i, :, ch[name]], fs, name.lower()))
        if "RC" in ch and "RA" in ch:
            f.update(coupling_features(x[i, :, ch["RC"]], x[i, :, ch["RA"]]))
        if "HRV" in ch:
            h = x[i, :, ch["HRV"]]
            f["hrv_chan_mean"], f["hrv_chan_std"] = float(h.mean()), float(h.std())
        rows.append(f)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Separability analysis
# ---------------------------------------------------------------------------
def pairwise_feature_auc(feats: pd.DataFrame, y: np.ndarray, pairs: Sequence[Tuple[int, int]] = ((1, 0), (1, 2), (1, 4), (0, 2), (2, 3)),
                         class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    """Univariate AUC (folded to >= 0.5) of every feature for each stage pair."""
    out = {}
    for a, b in pairs:
        m = (y == a) | (y == b)
        col = {}
        for f in feats.columns:
            v = feats.loc[m, f].to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 20 or len(np.unique(y[m][ok])) < 2:
                col[f] = np.nan
                continue
            auc = roc_auc_score((y[m][ok] == a).astype(int), v[ok])
            col[f] = max(auc, 1 - auc)
        out[f"{class_names[a]} vs {class_names[b]}"] = col
    return pd.DataFrame(out)


def pairwise_feature_classifier(feats: pd.DataFrame, y: np.ndarray, subject: Optional[np.ndarray] = None,
                                pairs: Sequence[Tuple[int, int]] = ((1, 0), (1, 2), (1, 4), (0, 2), (2, 3), (2, 4)),
                                class_names: Sequence[str] = CLASS_NAMES, seed: int = 0,
                                min_per_class: int = 20) -> pd.DataFrame:
    """Balanced accuracy / AUC of a logistic regression on *all* features for each stage pair (subject-wise CV).

    Pairs with fewer than ``min_per_class`` epochs of either stage are reported as NaN."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import balanced_accuracy_score

    rows = []
    X_all = feats.to_numpy(dtype=float)
    for a, b in pairs:
        m = (y == a) | (y == b)
        X, t = X_all[m], (y[m] == a).astype(int)
        row = {"pair": f"{class_names[a]} vs {class_names[b]}", "n": int(m.sum()),
               "n_pos": int(t.sum()), "auc": np.nan, "balanced_acc": np.nan}
        if min_per_class > 0 and (t.sum() < min_per_class or (len(t) - t.sum()) < min_per_class):
            rows.append(row)          # too few epochs of one stage to estimate separability
            continue
        clf = make_pipeline(SimpleImputer(), StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        n_groups = len(np.unique(subject[m])) if subject is not None else 0
        try:
            if n_groups >= 2:
                cv = GroupKFold(n_splits=min(5, n_groups))
                p = cross_val_predict(clf, X, t, groups=subject[m], cv=cv, method="predict_proba")[:, 1]
            else:
                cv = StratifiedKFold(max(2, min(5, int(np.bincount(t).min()))), shuffle=True, random_state=seed)
                p = cross_val_predict(clf, X, t, cv=cv, method="predict_proba")[:, 1]
            row.update(auc=roc_auc_score(t, p), balanced_acc=balanced_accuracy_score(t, p > 0.5))
        except ValueError:            # e.g. a subject-wise fold that contains a single class (tiny data sets)
            pass
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


def plot_feature_distributions(feats: pd.DataFrame, y: np.ndarray, features: Sequence[str],
                               class_names: Sequence[str] = CLASS_NAMES, ncols: int = 3):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(features)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    df = feats.copy()
    df["stage"] = [class_names[k] for k in y]
    for ax, f in zip(axes.ravel(), features):
        sns.violinplot(data=df, x="stage", y=f, order=list(class_names), inner="quartile", cut=0, ax=ax, density_norm="width")
        ax.set_xlabel("")
        ax.set_title(f)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig
