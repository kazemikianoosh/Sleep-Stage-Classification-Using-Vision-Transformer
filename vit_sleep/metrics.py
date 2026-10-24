"""Evaluation metrics, subject-level bootstrap confidence intervals and table/figure helpers.

No constant offsets, no post-hoc adjustments: every number reported by this module
is computed directly from ``y_true`` / ``y_pred`` / ``probs`` with scikit-learn.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import CLASS_NAMES, NUM_CLASSES


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: Optional[np.ndarray] = None,
                    class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, float]:
    """Accuracy, weighted/macro precision-recall-F1, per-class F1, Cohen's kappa (+ per-class AUC)."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    labels = list(range(len(class_names)))
    out: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
    }
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for k, name in enumerate(class_names):
        out[f"f1_{name}"] = float(f1s[k])
    if probs is not None:
        probs = np.asarray(probs)
        for k, name in enumerate(class_names):
            pos = (y_true == k).astype(int)
            out[f"auc_{name}"] = roc_auc_score(pos, probs[:, k]) if 0 < pos.sum() < len(pos) else np.nan
    return {k: float(v) for k, v in out.items()}


def per_class_report(y_true, y_pred, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    df = pd.DataFrame({
        "precision": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "support": np.bincount(np.asarray(y_true), minlength=len(class_names)),
    }, index=list(class_names))
    return df


def confusion_df(y_true, y_pred, normalize: Optional[str] = "true",
                 class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize=normalize)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])



def roc_curves(y_true, probs, class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, dict]:
    out = {}
    for k, name in enumerate(class_names):
        pos = (np.asarray(y_true) == k).astype(int)
        fpr, tpr, _ = roc_curve(pos, probs[:, k])
        out[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(pos, probs[:, k])}
    return out


def entropy_confidence(probs: np.ndarray) -> np.ndarray:
    """1 - H(p)/log(K) per epoch (Section 4.7 of the paper)."""
    p = np.clip(np.asarray(probs), 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=-1)
    return 1.0 - h / np.log(p.shape[-1])


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, subject: Optional[np.ndarray] = None,
                 metric: Callable = None, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                 probs: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Percentile bootstrap CI for every key returned by ``metric`` (default :func:`compute_metrics`).

    When ``subject`` is given, *subjects* (recordings) are resampled with replacement -
    the appropriate unit for an inter-patient protocol; otherwise epochs are resampled.
    """
    metric = metric or compute_metrics
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    point = metric(y_true, y_pred, probs) if probs is not None else metric(y_true, y_pred)
    samples: Dict[str, List[float]] = {k: [] for k in point}
    if subject is not None:
        subject = np.asarray(subject)
        groups = [np.where(subject == s)[0] for s in np.unique(subject)]
    for _ in range(n_boot):
        if subject is not None:
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
        else:
            idx = rng.integers(0, len(y_true), len(y_true))
        m = metric(y_true[idx], y_pred[idx], probs[idx]) if probs is not None else metric(y_true[idx], y_pred[idx])
        for k, v in m.items():
            samples[k].append(v)
    rows = []
    for k, v in point.items():
        arr = np.asarray(samples[k], dtype=float)
        rows.append({"metric": k, "value": v,
                     "ci_low": np.nanpercentile(arr, 100 * alpha / 2),
                     "ci_high": np.nanpercentile(arr, 100 * (1 - alpha / 2))})
    return pd.DataFrame(rows).set_index("metric")


def paired_bootstrap_delta(y_true, pred_a, pred_b, subject=None, key: str = "kappa", n_boot: int = 1000,
                           seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap of ``metric_b - metric_a`` (same resampled subjects for both models)."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    if subject is not None:
        subject = np.asarray(subject)
        groups = [np.where(subject == s)[0] for s in np.unique(subject)]
    deltas = []
    for _ in range(n_boot):
        if subject is not None:
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
        else:
            idx = rng.integers(0, len(y_true), len(y_true))
        deltas.append(compute_metrics(y_true[idx], pred_b[idx])[key] - compute_metrics(y_true[idx], pred_a[idx])[key])
    d = np.asarray(deltas)
    return {"delta": float(compute_metrics(y_true, pred_b)[key] - compute_metrics(y_true, pred_a)[key]),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def per_subject_metrics(y_true, y_pred, subject, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    rows = []
    for s in np.unique(subject):
        m = subject == s
        r = compute_metrics(y_true[m], y_pred[m], class_names=class_names)
        r["subject"] = s
        r["n_epochs"] = int(m.sum())
        for k, name in enumerate(class_names):
            r[f"prev_{name}"] = float((y_true[m] == k).mean())
        rows.append(r)
    return pd.DataFrame(rows).set_index("subject")


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def format_ci(row: pd.Series, digits: int = 2) -> str:
    return f"{row['value']:.{digits}f} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"


def results_table(results: Dict[str, Dict[str, float]], keys: Iterable[str] = (
        "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "kappa",
        "f1_Wake", "f1_N1", "f1_N2", "f1_N3", "f1_REM")) -> pd.DataFrame:
    """``{config_name: metrics_dict}`` -> tidy DataFrame (rows = configs)."""
    return pd.DataFrame({name: {k: m.get(k, np.nan) for k in keys} for name, m in results.items()}).T


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "", digits: int = 2) -> str:
    return df.to_latex(float_format=lambda v: f"{v:.{digits}f}", caption=caption, label=label, escape=True)


def plot_confusion_matrix(y_true, y_pred, class_names: Sequence[str] = CLASS_NAMES, ax=None, title: str = ""):
    """Row-normalised confusion matrix annotated with counts and percentages."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    perc = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    annot = np.array([[f"{cm[i, j]}\n{perc[i, j]:.1f}%" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(perc, annot=annot, fmt="", cmap="Blues", cbar=False, square=True, ax=ax,
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference (PSG)")
    ax.set_title(title)
    return ax
