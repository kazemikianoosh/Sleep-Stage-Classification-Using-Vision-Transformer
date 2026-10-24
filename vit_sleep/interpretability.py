"""Attention-based explanations (Section 2.5 / 4.4 of the paper) and per-channel attribution.

Important change w.r.t. the legacy notebooks
--------------------------------------------
Every token produced by the Conv1D patch embedding mixes *all* input channels, so the
attention that the [CLS] token pays to a patch cannot be split into per-channel shares by
striding over the attention vector (as the legacy ``get_normalized_mean_attention_scores``
did). Because the embedding is linear, however, each token decomposes exactly into one
contribution per channel::

    z_i = b + sum_c  v_{i,c},      v_{i,c} = sum_p W[p, c, :] * x[iP + p, c]

:func:`channel_token_contributions` computes ``||v_{i,c}||`` and
:func:`attention_weighted_channel_importance` weights it by the [CLS] attention of the
patch. Two model-agnostic views are provided as sanity checks:
:func:`occlusion_channel_importance` (performance drop when a channel is blanked) and
:func:`gradient_channel_importance` (|gradient x input| mass per channel).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

from . import CHANNEL_NAMES, CLASS_NAMES, EPOCH_SECONDS
from .metrics import compute_metrics


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------
def cls_attention(model, x: np.ndarray, block: int = -1, batch_size: int = 128) -> np.ndarray:
    """[CLS] -> patch attention of one block, ``(B, H, N)`` (Eq. 9)."""
    out = []
    for s in range(0, len(x), batch_size):
        xb = tf.constant(x[s:s + batch_size], tf.float32)
        a = model.get_all_selfattention(xb)[block] if block != -1 else model.get_last_selfattention(xb)
        out.append(a[:, :, 0, 1:].numpy())
    return np.concatenate(out, axis=0)


def attention_rollout(model, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020): [CLS] row of prod_l (A_l + I)/2 averaged over heads -> ``(B, N)``."""
    out = []
    for s in range(0, len(x), batch_size):
        xb = tf.constant(x[s:s + batch_size], tf.float32)
        mats = model.get_all_selfattention(xb)
        n = mats[0].shape[-1]
        roll = tf.eye(n, batch_shape=tf.shape(xb)[:1])
        for a in mats:
            a = tf.reduce_mean(a, axis=1)                         # head fusion
            a = 0.5 * (a + tf.eye(n))
            a = a / tf.reduce_sum(a, axis=-1, keepdims=True)
            roll = tf.matmul(a, roll)
        out.append(roll[:, 0, 1:].numpy())
    return np.concatenate(out, axis=0)


def upsample_attention(attn: np.ndarray, seq_len: int) -> np.ndarray:
    """``(..., N)`` patch attention -> ``(..., seq_len)`` by repeating each patch's value."""
    n = attn.shape[-1]
    patch = seq_len // n
    up = np.repeat(attn, patch, axis=-1)
    if up.shape[-1] < seq_len:
        pad = seq_len - up.shape[-1]
        up = np.concatenate([up, np.repeat(up[..., -1:], pad, axis=-1)], axis=-1)
    return up[..., :seq_len]


def attention_entropy(attn: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalised entropy (0 = one patch, 1 = uniform) of a ``(..., N)`` attention distribution."""
    p = attn / np.maximum(attn.sum(axis=-1, keepdims=True), eps)
    h = -(p * np.log(p + eps)).sum(axis=-1)
    return h / np.log(attn.shape[-1])


def head_projection_scores(model, block: int = -1) -> Dict[str, np.ndarray]:
    """Per-head statistics of the MHSA output projection W_O ``(H, d_k, D)`` (Tables 3-4 of the paper).

    ``mean_minmax`` reproduces the paper's normalised mean; ``abs_mean`` / ``frobenius``
    are sign-invariant alternatives that are more defensible as head-importance scores.
    """
    W = model.get_last_projection_matrix(block)
    mean = W.mean(axis=(1, 2))
    return {
        "mean_minmax": (mean - mean.min()) / max(mean.max() - mean.min(), 1e-12),
        "abs_mean": np.abs(W).mean(axis=(1, 2)),
        "frobenius": np.sqrt((W ** 2).sum(axis=(1, 2))),
    }


# ---------------------------------------------------------------------------
# Per-channel attribution
# ---------------------------------------------------------------------------
def channel_token_contributions(model, x: np.ndarray) -> np.ndarray:
    """``||v_{i,c}||_2`` for every token i and channel c -> ``(B, N, C)``."""
    W = model.patch_kernel()                            # (P, C, D)
    P = W.shape[0]
    B, T, C = x.shape
    N = T // P
    patches = x[:, : N * P].reshape(B, N, P, C)          # (B, N, P, C)
    v = np.einsum("bnpc,pcd->bncd", patches.astype(np.float32), W)   # (B, N, C, D)
    return np.linalg.norm(v, axis=-1)


def attention_weighted_channel_importance(model, x: np.ndarray, attn: Optional[np.ndarray] = None,
                                          use_rollout: bool = False, batch_size: int = 128,
                                          channels: Sequence[str] = None) -> Dict[str, np.ndarray]:
    """Share of [CLS]-attended token energy attributable to each input channel.

    Returns per-epoch shares ``(B, C)`` (rows sum to 1) and their mean.
    """
    if attn is None:
        attn = attention_rollout(model, x, batch_size) if use_rollout else cls_attention(model, x, batch_size=batch_size).mean(axis=1)
    contrib = channel_token_contributions(model, x)                    # (B, N, C)
    share = contrib / np.maximum(contrib.sum(axis=-1, keepdims=True), 1e-12)   # per-token channel shares
    w = attn / np.maximum(attn.sum(axis=-1, keepdims=True), 1e-12)     # (B, N)
    per_epoch = np.einsum("bn,bnc->bc", w, share)
    names = list(channels) if channels is not None else [f"ch{c}" for c in range(x.shape[-1])]
    return {"per_epoch": per_epoch, "mean": per_epoch.mean(axis=0), "channels": names}


def occlusion_channel_importance(model, x: np.ndarray, y: np.ndarray, channels: Sequence[str],
                                 mode: str = "mean", batch_size: int = 256) -> Dict[str, Dict[str, float]]:
    """Performance drop when one channel is replaced by a constant (its per-epoch mean) or shuffled across epochs."""
    def _probs(arr):
        return tf.nn.softmax(model.predict(arr, batch_size=batch_size, verbose=0), axis=-1).numpy()

    base = _probs(x)
    base_m = compute_metrics(y, base.argmax(1))
    base_conf = base[np.arange(len(y)), y].mean()
    out = {"none": {"accuracy": base_m["accuracy"], "kappa": base_m["kappa"], "true_class_prob": float(base_conf),
                    "d_accuracy": 0.0, "d_kappa": 0.0, "d_true_class_prob": 0.0}}
    rng = np.random.default_rng(0)
    for c, name in enumerate(channels):
        xo = x.copy()
        if mode == "mean":
            xo[..., c] = xo[..., c].mean(axis=1, keepdims=True)
        elif mode == "zero":
            xo[..., c] = 0.0
        elif mode == "shuffle":
            xo[..., c] = xo[rng.permutation(len(xo)), :, c]
        else:
            raise ValueError(mode)
        p = _probs(xo)
        m = compute_metrics(y, p.argmax(1))
        conf = p[np.arange(len(y)), y].mean()
        out[name] = {"accuracy": m["accuracy"], "kappa": m["kappa"], "true_class_prob": float(conf),
                     "d_accuracy": base_m["accuracy"] - m["accuracy"], "d_kappa": base_m["kappa"] - m["kappa"],
                     "d_true_class_prob": float(base_conf - conf)}
    return out


def gradient_channel_importance(model, x: np.ndarray, y: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Mean |gradient x input| mass per channel (normalised to sum 1) for the true-class logit -> ``(C,)``."""
    tot = np.zeros(x.shape[-1])
    for s in range(0, len(x), batch_size):
        xb = tf.constant(x[s:s + batch_size], tf.float32)
        yb = tf.constant(y[s:s + batch_size])
        with tf.GradientTape() as tape:
            tape.watch(xb)
            logits = model(xb, training=False)
            target = tf.gather(logits, yb, axis=1, batch_dims=1)
        g = tape.gradient(target, xb)
        tot += tf.reduce_sum(tf.abs(g * xb), axis=(0, 1)).numpy()
    return tot / max(tot.sum(), 1e-12)


# ---------------------------------------------------------------------------
# Attention on physiological landmarks (Table 5 of the paper)
# ---------------------------------------------------------------------------
def landmark_attention_share(signal: np.ndarray, attn_1d: np.ndarray, fs: float, kind: str = "ppg",
                             halfwidth_s: float = 0.08) -> Dict[str, float]:
    """Fraction of attention mass inside +-``halfwidth_s`` of peaks / troughs of ``signal``.

    ``kind="ppg"`` uses a 0.33-s minimum peak distance (systolic peaks), ``kind="resp"`` 1.5 s.
    The remaining mass is reported as ``other``. Uniform attention would give
    ``2*halfwidth_s*n_events/epoch_len`` - also returned for reference.
    """
    sig = np.asarray(signal, dtype=float)
    dist = 0.33 if kind == "ppg" else 1.5
    pk, _ = find_peaks(sig - np.median(sig), distance=int(dist * fs), prominence=0.25 * sig.std() + 1e-9)
    tr, _ = find_peaks(-(sig - np.median(sig)), distance=int(dist * fs), prominence=0.25 * sig.std() + 1e-9)
    w = attn_1d / max(attn_1d.sum(), 1e-12)
    hw = max(1, int(halfwidth_s * fs))
    T = len(sig)

    def mass(idx):
        m = np.zeros(T, dtype=bool)
        for i in idx:
            m[max(0, i - hw):min(T, i + hw + 1)] = True
        return float(w[m].sum()), float(m.mean())

    p_mass, p_cov = mass(pk)
    t_mass, t_cov = mass(tr)
    return {"peaks": p_mass, "troughs": t_mass, "other": max(0.0, 1 - p_mass - t_mass),
            "peaks_uniform_baseline": p_cov, "troughs_uniform_baseline": t_cov, "n_peaks": len(pk)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_attention_over_signal(ax, signal: np.ndarray, attn_1d: np.ndarray, epoch_seconds: int = EPOCH_SECONDS,
                               cmap: str = "Reds", alpha: float = 0.6, title: str = "", ylabel: str = "Amplitude"):
    """Signal trace with attention as background heat-map (Figs. 5-6 of the paper)."""
    t = np.linspace(0, epoch_seconds, len(signal))
    ax.plot(t, signal, lw=0.7, color=(0, 0, 0.7))
    ax.margins(0)
    ax.set_ylim(signal.min(), signal.max())
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(), attn_1d[np.newaxis], cmap=cmap, alpha=alpha)
    ax.set_ylabel(ylabel)
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_all_heads(model, x_epoch: np.ndarray, channel_idx: int = 0, channel_name: str = "PPG",
                   epoch_seconds: int = EPOCH_SECONDS):
    import matplotlib.pyplot as plt

    attn = cls_attention(model, x_epoch[None])[0]                        # (H, N)
    up = upsample_attention(attn, x_epoch.shape[0])
    H = attn.shape[0]
    fig, axes = plt.subplots(H, 1, figsize=(11, 1.6 * H), sharex=True)
    for h in range(H):
        plot_attention_over_signal(axes[h], x_epoch[:, channel_idx], up[h], epoch_seconds,
                                   title=f"{channel_name} - head {h + 1}")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
