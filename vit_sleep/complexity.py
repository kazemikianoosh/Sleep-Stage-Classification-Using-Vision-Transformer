"""Model complexity: parameter counts, analytic FLOPs of the ViT-1D, measured latency.

Used by the patch-length / token-length ablation (Reviewer comment 5) to show the
accuracy-vs-cost trade-off, and by the baseline comparison (comment 4).
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np


def count_params(model) -> int:
    return int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def vit_flops(seq_len: int, n_channels: int, patch_size: int, hidden: int, depth: int, num_heads: int,
              mlp_dim: int, num_classes: int = 5, patch_stride: int | None = None) -> Dict[str, float]:
    """Multiply-accumulate count (1 MAC = 2 FLOPs) of one forward pass of the ViT-1D.

    Returns the split between patch embedding, attention (QKV + scores + context + output
    projection) and the MLP so that the O(N^2 D) part can be discussed explicitly.
    """
    stride = patch_size if patch_stride is None else patch_stride
    n_patches = (seq_len - patch_size) // stride + 1
    N = n_patches + 1                                  # tokens incl. [CLS]
    embed = n_patches * patch_size * n_channels * hidden
    qkv = 3 * N * hidden * hidden
    scores = num_heads * N * N * (hidden // num_heads)   # Q K^T
    context = num_heads * N * N * (hidden // num_heads)  # A V
    out_proj = N * hidden * hidden
    mlp = 2 * N * hidden * mlp_dim
    attn_total = depth * (qkv + scores + context + out_proj)
    mlp_total = depth * mlp
    head = hidden * num_classes
    total = embed + attn_total + mlp_total + head
    return {"tokens": N, "macs_embed": embed, "macs_attention": attn_total, "macs_attention_quadratic": depth * (scores + context),
            "macs_mlp": mlp_total, "macs_total": total, "gflops": 2 * total / 1e9}


def measure_latency(model, x: np.ndarray, n_repeats: int = 100, batch_size: int = 1, warmup: int = 10) -> Dict[str, float]:
    """Mean / std wall-clock inference time per batch (seconds) on the current device."""
    import tensorflow as tf

    xb = tf.constant(x[:batch_size], tf.float32)
    fn = tf.function(lambda t: model(t, training=False))
    for _ in range(warmup):
        fn(xb)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = fn(xb).numpy()
        times.append(time.perf_counter() - t0)
    return {"latency_mean_s": float(np.mean(times)), "latency_std_s": float(np.std(times)), "batch_size": batch_size}
