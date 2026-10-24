"""ViT-Sleep: interpretable five-stage sleep classification from cardiorespiratory signals.

Sub-modules
-----------
config            experiment configuration dataclasses
seed              reproducibility helpers
data              loading, resampling, normalisation, channel selection, tf.data pipelines, sequence windows
features          hand-crafted autonomic features (HR/HRV/respiration) used for the N1 error analysis
metrics           accuracy / F1 / kappa / AUC, subject-level bootstrap CIs, confusion matrices, LaTeX export
transitions       hypnogram transition statistics, HMM (Viterbi) smoothing, transitional-epoch masks
interpretability  CLS attention maps, attention rollout, per-channel attribution, channel occlusion
complexity        parameter counts, analytic FLOPs for the ViT, latency measurement
train             compile/fit helpers (LR schedule, early stopping, checkpointing)
models            ViT-1D, temporal-context (sequence) models and all baselines
"""

__version__ = "1.0.0"

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
CHANNEL_NAMES = ["PPG", "RF", "RC", "RA", "HRV"]
NUM_CLASSES = len(CLASS_NAMES)
EPOCH_SECONDS = 30
