"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 0, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow.

    Parameters
    ----------
    seed
        Random seed shared by all libraries.
    deterministic_ops
        If True, also request deterministic GPU kernels (``TF_DETERMINISTIC_OPS=1``).
        This slows training down and is off by default; results in the paper were
        obtained with fixed seeds but without op-level determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if hasattr(tf.keras.utils, "set_random_seed"):
            tf.keras.utils.set_random_seed(seed)
        if deterministic_ops:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
            if hasattr(tf.config.experimental, "enable_op_determinism"):
                tf.config.experimental.enable_op_determinism()
    except ImportError:  # pure-numpy analyses do not need TensorFlow
        pass
