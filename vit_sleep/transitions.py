"""Sleep-stage transition dynamics (Reviewer comment 1) and transitional-epoch analysis (comment 3).

* :func:`transition_matrix` - first-order stage-to-stage transition probabilities of the
  training hypnograms (the "macrostructural" prior).
* :func:`viterbi_decode` / :func:`hmm_smooth` - combine epoch-wise posteriors of *any*
  classifier with that prior (HMM with the classifier as emission model, scaled by the
  class prior). No re-training is required, so this isolates the value of modelling
  temporal progression.
* :func:`transition_mask` / :func:`context_breakdown` - split epochs into *stable* and
  *transitional* ones and report per-group performance, e.g. N1 preceded by Wake vs by N2.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import CLASS_NAMES, NUM_CLASSES
from .data import subject_runs
from .metrics import compute_metrics


# ---------------------------------------------------------------------------
# Transition statistics
# ---------------------------------------------------------------------------
def transition_matrix(labels: np.ndarray, subject: np.ndarray, smoothing: float = 1.0,
                      n_classes: int = NUM_CLASSES) -> np.ndarray:
    """Row-stochastic ``P[i, j] = P(stage_{t+1}=j | stage_t=i)`` with additive smoothing."""
    counts = np.full((n_classes, n_classes), smoothing, dtype=float)
    for run in subject_runs(subject):
        y = labels[run]
        np.add.at(counts, (y[:-1], y[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def transition_matrix_df(P: np.ndarray, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    return pd.DataFrame(P, index=[f"from_{c}" for c in class_names], columns=[f"to_{c}" for c in class_names])


def stage_prior(labels: np.ndarray, n_classes: int = NUM_CLASSES) -> np.ndarray:
    return np.bincount(labels, minlength=n_classes) / len(labels)


def bout_lengths(labels: np.ndarray, subject: np.ndarray) -> pd.DataFrame:
    """Length (in epochs) of every contiguous bout of each stage - shows how transient N1 is."""
    rows = []
    for run in subject_runs(subject):
        y = labels[run]
        start = 0
        for i in range(1, len(y) + 1):
            if i == len(y) or y[i] != y[start]:
                rows.append({"stage": CLASS_NAMES[y[start]], "length": i - start})
                start = i
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HMM smoothing of posteriors
# ---------------------------------------------------------------------------
def viterbi_decode(log_emission: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray) -> np.ndarray:
    """Most probable state path for one recording. ``log_emission`` is ``(T, K)``."""
    T, K = log_emission.shape
    delta = np.empty((T, K))
    back = np.zeros((T, K), dtype=np.int64)
    delta[0] = log_init + log_emission[0]
    for t in range(1, T):
        score = delta[t - 1][:, None] + log_trans          # (K_prev, K)
        back[t] = np.argmax(score, axis=0)
        delta[t] = score[back[t], np.arange(K)] + log_emission[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def forward_backward(log_emission: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray) -> np.ndarray:
    """Posterior marginals ``(T, K)`` (log-space forward-backward) - smoothed probabilities."""
    from scipy.special import logsumexp

    T, K = log_emission.shape
    fwd = np.empty((T, K))
    bwd = np.zeros((T, K))
    fwd[0] = log_init + log_emission[0]
    for t in range(1, T):
        fwd[t] = logsumexp(fwd[t - 1][:, None] + log_trans, axis=0) + log_emission[t]
    for t in range(T - 2, -1, -1):
        bwd[t] = logsumexp(log_trans + (log_emission[t + 1] + bwd[t + 1])[None, :], axis=1)
    post = fwd + bwd
    post -= logsumexp(post, axis=1, keepdims=True)
    return np.exp(post)


def hmm_smooth(probs: np.ndarray, subject: np.ndarray, P: np.ndarray, prior: Optional[np.ndarray] = None,
               init: Optional[np.ndarray] = None, mode: str = "viterbi", temperature: float = 1.0,
               prior_power: float = 0.0, eps: float = 1e-8) -> np.ndarray:
    """Apply first-order transition dynamics to epoch-wise class probabilities.

    Emission score ``log p(y|x)/T - prior_power * log p(y)``: ``prior_power=1`` is the classic
    hybrid-HMM scaled likelihood, ``prior_power=0`` (default) uses the posterior directly,
    which is more robust for heavily imbalanced stages such as N1. Tune ``temperature`` and
    ``prior_power`` on the validation split with :func:`tune_hmm`.

    ``mode="viterbi"`` returns hard labels; ``mode="posterior"`` returns smoothed probabilities ``(N, K)``.
    """
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    K = probs.shape[1]
    prior = np.ones(K) / K if prior is None else np.asarray(prior)
    init = prior if init is None else np.asarray(init)
    log_trans = np.log(np.clip(P, eps, 1.0))
    log_init = np.log(np.clip(init, eps, 1.0))
    log_em = np.log(probs) / temperature - prior_power * np.log(np.clip(prior, eps, 1.0))
    out = np.empty(len(probs), dtype=np.int64) if mode == "viterbi" else np.empty_like(probs)
    for run in subject_runs(subject):
        if mode == "viterbi":
            out[run] = viterbi_decode(log_em[run], log_trans, log_init)
        else:
            out[run] = forward_backward(log_em[run], log_trans, log_init)
    return out


def tune_hmm(probs_val, y_val, subject_val, P, prior, temperatures=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
             prior_powers=(0.0, 0.5, 1.0), key: str = "kappa") -> Tuple[Dict[str, float], pd.DataFrame]:
    """Choose emission temperature and prior power on the *validation* split (never on test)."""
    rows = []
    for pp in prior_powers:
        for t in temperatures:
            pred = hmm_smooth(probs_val, subject_val, P, prior, temperature=t, prior_power=pp)
            rows.append({"temperature": t, "prior_power": pp, **compute_metrics(y_val, pred)})
    df = pd.DataFrame(rows)
    best = df.loc[df[key].idxmax()]
    return {"temperature": float(best["temperature"]), "prior_power": float(best["prior_power"])}, df.set_index(["prior_power", "temperature"])


HMM_DEFAULTS = {"temperature": 1.0, "prior_power": 0.0}


def tune_hmm_or_default(probs_val, y_val, seq_val, P, prior, **kw) -> Tuple[Dict[str, float], Optional[pd.DataFrame], str]:
    """Tune on validation when it has a chronology (``seq_val`` not None), otherwise fall back to
    :data:`HMM_DEFAULTS` without tuning (a random epoch subset cannot be Viterbi-decoded)."""
    if seq_val is None or probs_val is None:
        return dict(HMM_DEFAULTS), None, "validation split has no chronology -> HMM hyper-parameters left at defaults (T=1, prior^0), no tuning"
    best, grid = tune_hmm(probs_val, y_val, seq_val, P, prior, **kw)
    return best, grid, f"HMM hyper-parameters tuned on validation: {best}"


def tune_hmm_temperature(probs_val, y_val, subject_val, P, prior, grid=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                         key: str = "kappa") -> Tuple[float, pd.DataFrame]:
    """Backwards-compatible wrapper: temperature only (prior_power = 0)."""
    best, df = tune_hmm(probs_val, y_val, subject_val, P, prior, grid, (0.0,), key)
    return best["temperature"], df.reset_index(level=0, drop=True)


# ---------------------------------------------------------------------------
# Stable vs transitional epochs
# ---------------------------------------------------------------------------
def transition_mask(labels: np.ndarray, subject: np.ndarray, radius: int = 1) -> np.ndarray:
    """True for epochs within ``radius`` epochs of a stage change (incl. both sides of the change)."""
    labels = np.asarray(labels)
    mask = np.zeros(len(labels), dtype=bool)
    for run in subject_runs(subject):
        y = labels[run]
        change = np.where(np.diff(y) != 0)[0]                # change between i and i+1
        for c in change:
            lo, hi = max(0, c - radius + 1), min(len(y), c + radius + 1)
            mask[run[lo:hi]] = True
    return mask


def previous_stage(labels: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """Stage of the last epoch with a *different* label (-1 if none) - e.g. what N1 was entered from."""
    labels = np.asarray(labels)
    prev = -np.ones(len(labels), dtype=np.int64)
    for run in subject_runs(subject):
        y = labels[run]
        last_diff = -1
        for i in range(len(y)):
            if i > 0 and y[i] != y[i - 1]:
                last_diff = y[i - 1]
            prev[run[i]] = last_diff
    return prev


def context_breakdown(y_true, y_pred, subject, radius: int = 1,
                      class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    """Metrics on stable vs transitional epochs, overall and per reference stage (recall)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    trans = transition_mask(y_true, subject, radius)
    rows = []
    for name, m in (("all", np.ones_like(trans)), ("stable", ~trans), ("transitional", trans)):
        r = {"subset": name, "n": int(m.sum()), **{k: v for k, v in compute_metrics(y_true[m], y_pred[m]).items()
                                                  if k in ("accuracy", "kappa", "f1_weighted", "f1_macro")}}
        for k, c in enumerate(class_names):
            mm = m & (y_true == k)
            r[f"recall_{c}"] = float((y_pred[mm] == k).mean()) if mm.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("subset")


def n1_by_previous_stage(y_true, y_pred, subject, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    """For reference-N1 epochs: recall and predicted-label distribution grouped by the stage N1 was entered from."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    prev = previous_stage(y_true, subject)
    is_n1 = y_true == 1
    rows = []
    for p in [-1] + list(range(len(class_names))):
        m = is_n1 & (prev == p)
        if not m.any():
            continue
        dist = np.bincount(y_pred[m], minlength=len(class_names)) / m.sum()
        rows.append({"entered_from": "start" if p == -1 else class_names[p], "n": int(m.sum()),
                     "recall_N1": float((y_pred[m] == 1).mean()),
                     **{f"pred_{c}": float(dist[k]) for k, c in enumerate(class_names)}})
    return pd.DataFrame(rows).set_index("entered_from")


def n1_by_bout_position(y_true, y_pred, subject, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    """Recall of N1 as a function of bout length (isolated 1-epoch N1 vs sustained N1 runs)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for run in subject_runs(subject):
        y = y_true[run]
        start = 0
        for i in range(1, len(y) + 1):
            if i == len(y) or y[i] != y[start]:
                if y[start] == 1:
                    idx = run[start:i]
                    rows.append({"bout_len": i - start, "correct": int((y_pred[idx] == 1).sum()), "n": i - start})
                start = i
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    bins = pd.cut(df["bout_len"], [0, 1, 2, 4, 8, np.inf], labels=["1", "2", "3-4", "5-8", ">8"])
    g = df.groupby(bins, observed=True)[["correct", "n"]].sum()
    g["recall_N1"] = g["correct"] / g["n"]
    g["n_bouts"] = df.groupby(bins, observed=True).size()
    return g
