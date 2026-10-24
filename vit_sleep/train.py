"""Training / evaluation loop shared by the CLI scripts and the notebook."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from . import CLASS_NAMES, NUM_CLASSES
from .complexity import count_params, vit_flops
from .config import ExperimentConfig
from .data import (SplitData, class_weights, gather_windows, load_experiment_data, make_context_windows,
                   make_dataset)
from .metrics import compute_metrics
from .models import build_context_model, build_model
from .seed import set_seed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
def configure_gpu(memory_growth: bool = True, mixed_precision: bool = False) -> list:
    """Enable memory growth on all visible GPUs (call once, before any model is built).

    TensorFlow uses a GPU automatically when one is visible; this only prevents it from
    reserving the whole card up-front, which matters on shared machines. Returns the GPU list.

    ``mixed_precision=True`` sets the global Keras policy to ``mixed_float16`` (1.5-3x faster on
    V100/A100-class GPUs with tensor cores; all classification heads keep float32 logits).
    It is ignored when no GPU is visible.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if memory_growth:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except (RuntimeError, ValueError):   # already initialised
                pass
    if mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("mixed precision enabled (mixed_float16)")
    elif mixed_precision:
        log.warning("mixed precision requested but no GPU visible - keeping float32")
    log.info("visible GPUs: %s", [g.name for g in gpus] or "none (CPU)")
    return gpus


# ---------------------------------------------------------------------------
# Compile / fit helpers
# ---------------------------------------------------------------------------
def make_optimizer(name: str, lr: float, weight_decay: float = 0.0):
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(name)


def compile_model(model, lr: float = 1e-4, optimizer: str = "adamw", weight_decay: float = 1e-4,
                  label_smoothing: float = 0.0):
    model.compile(
        optimizer=make_optimizer(optimizer, lr, weight_decay),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model


def default_callbacks(out_dir: Path, lr: float, lr_drop_epoch: Optional[int] = None,
                      early_stopping_patience: Optional[int] = None, monitor: str = "val_loss"):
    out_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(str(out_dir / "best.weights.h5"), monitor=monitor,
                                           save_best_only=True, save_weights_only=True, verbose=0),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]
    if lr_drop_epoch:
        cbs.append(tf.keras.callbacks.LearningRateScheduler(lambda ep, cur: lr if ep < lr_drop_epoch else lr / 10))
    if early_stopping_patience:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=early_stopping_patience,
                                                    restore_best_weights=False))
    return cbs


def sample_weights_from_class_weight(y: np.ndarray, mode: Optional[str]) -> Optional[np.ndarray]:
    cw = class_weights(y, mode)
    if cw is None:
        return None
    return np.asarray([cw[int(k)] for k in y], dtype=np.float32)


def predict_proba(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    logits = model.predict(x, batch_size=batch_size, verbose=0)
    return tf.nn.softmax(logits, axis=-1).numpy()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_predictions(out_dir: Path, split: str, probs: np.ndarray, sd: SplitData) -> None:
    np.savez_compressed(out_dir / f"pred_{split}.npz", probs=probs.astype(np.float32), y=sd.y,
                        subject=np.asarray(sd.subject if sd.subject is not None else np.full(len(sd.y), -1)),
                        disorder=np.asarray(sd.disorder if sd.disorder is not None else np.full(len(sd.y), "")),
                        sequence_id=np.asarray(sd.sequence_id if sd.sequence_id is not None else np.full(len(sd.y), -1)),
                        sequence_kind=np.asarray(sd.sequence_kind))


def load_predictions(out_dir: str | Path, split: str = "test") -> Dict[str, np.ndarray]:
    d = np.load(Path(out_dir) / f"pred_{split}.npz", allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["y_pred"] = out["probs"].argmax(1)
    subj = out["subject"]
    if subj.dtype.kind in "iu" and (subj == -1).all():
        out["subject"] = None           # saved without subject ids
    dis = out.get("disorder")
    if dis is not None and (dis.astype(str) == "").all():
        out["disorder"] = None
    sid = out.get("sequence_id")
    if sid is None or (sid.dtype.kind in "iu" and (sid == -1).all()):
        out["sequence_id"] = None
    out["seq"] = out["subject"] if out["subject"] is not None else out["sequence_id"]
    out["sequence_kind"] = str(out["sequence_kind"]) if "sequence_kind" in out else ("subject" if out["subject"] is not None else "none")
    return out


# ---------------------------------------------------------------------------
# Full epoch-level experiment
# ---------------------------------------------------------------------------
def experiment_exists(cfg: ExperimentConfig) -> bool:
    """True when ``results/<name>/`` already holds test predictions and a summary."""
    out_dir = Path(cfg.output_dir) / cfg.name
    return (out_dir / "pred_test.npz").exists() and (out_dir / "summary.json").exists()


def load_experiment_summary(cfg: ExperimentConfig, input_shape=None, load_model: bool = True) -> Dict:
    """Re-read a finished run's ``summary.json`` (and optionally rebuild its model) instead of retraining."""
    out_dir = Path(cfg.output_dir) / cfg.name
    summary = json.loads((out_dir / "summary.json").read_text())
    summary["model_obj"] = load_trained_model(out_dir, input_shape) if (load_model and input_shape is not None) else None
    summary["skipped"] = True
    return summary


def run_experiment(cfg: ExperimentConfig, data: Optional[Dict[str, SplitData]] = None,
                   verbose: int = 1, save: bool = True, skip_existing: bool = False) -> Dict:
    """Train one epoch-level model and evaluate it on val/test. Returns a summary dict.

    With ``skip_existing=True`` a run whose ``pred_test.npz`` + ``summary.json`` already exist
    in ``<output_dir>/<name>/`` is *not* retrained; its saved summary (and model) are returned.
    """
    set_seed(cfg.train.seed)
    out_dir = Path(cfg.output_dir) / cfg.name
    if data is None:
        data = load_experiment_data(cfg.data.data_dir, cfg.data.channels, cfg.data.target_len,
                                    cfg.data.normalisation, cfg.data.val_fraction, cfg.train.seed)
    train, val, test = data["train"], data["val"], data["test"]
    input_shape = train.x.shape[1:]

    if skip_existing and experiment_exists(cfg):
        log.info("%s exists - skipping training (use --force to retrain)", out_dir)
        return load_experiment_summary(cfg, input_shape)

    model = build_model(cfg.model, input_shape, NUM_CLASSES, **_model_hp(cfg))
    compile_model(model, cfg.train.learning_rate, cfg.train.optimizer, cfg.train.weight_decay, cfg.train.label_smoothing)

    sw = sample_weights_from_class_weight(train.y, cfg.train.class_weight)
    train_ds = make_dataset(train.x, train.y, cfg.data.batch_size, shuffle=True, seed=cfg.train.seed, sample_weight=sw)
    val_ds = make_dataset(val.x, val.y, cfg.data.batch_size)

    cbs = default_callbacks(out_dir, cfg.train.learning_rate, cfg.train.lr_drop_epoch, cfg.train.early_stopping_patience)
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.train.epochs, callbacks=cbs, verbose=verbose)
    train_time = time.time() - t0
    if (out_dir / "best.weights.h5").exists():
        model.load_weights(str(out_dir / "best.weights.h5"))    # best validation-loss checkpoint

    summary = {"name": cfg.name, "model": cfg.model, "channels": list(cfg.data.channels),
               "n_params": count_params(model), "train_time_s": train_time, "config": cfg.to_dict()}
    if cfg.model == "vit":
        summary["flops"] = vit_flops(input_shape[0], input_shape[1], cfg.vit.patch_size, cfg.vit.hidden_size,
                                     cfg.vit.depth, cfg.vit.num_heads, cfg.vit.mlp_dim,
                                     patch_stride=cfg.vit.patch_stride)
    for split, sd in (("val", val), ("test", test)):
        probs = predict_proba(model, sd.x)
        summary[f"metrics_{split}"] = compute_metrics(sd.y, probs.argmax(1), probs)
        if save:
            save_predictions(out_dir, split, probs, sd)
    if save:
        cfg.save(out_dir / "config.json")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        model.save_weights(str(out_dir / "final.weights.h5"))
    summary["model_obj"] = model
    summary["skipped"] = False
    log.info("%s | test acc %.3f kappa %.3f", cfg.name, summary["metrics_test"]["accuracy"], summary["metrics_test"]["kappa"])
    return summary


def _model_hp(cfg: ExperimentConfig) -> dict:
    if cfg.model == "vit":
        return dict(cfg.vit.__dict__)
    if cfg.model == "lite_transformer":
        return {"patch_size": cfg.vit.patch_size}
    return {}


def load_trained_model(out_dir: str | Path, input_shape) -> tf.keras.Model:
    """Rebuild any model from ``config.json`` and load its best weights."""
    out_dir = Path(out_dir)
    cfg = ExperimentConfig.load(out_dir / "config.json")
    candidates = [out_dir / "best.weights.h5", out_dir / "final.weights.h5"]
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        raise FileNotFoundError(f"no weights found in {out_dir}")

    failures = []
    for weights_path in candidates:
        # Rebuild for every attempt because a failed Keras load can partially
        # mutate a model. The dummy call explicitly materializes variables for
        # subclassed/custom layers under Keras 3.
        model = build_model(cfg.model, input_shape, NUM_CLASSES, **_model_hp(cfg))
        model(tf.zeros((1, *input_shape), tf.float32), training=False)
        try:
            model.load_weights(str(weights_path))
            if weights_path.name != "best.weights.h5":
                log.warning("%s: best checkpoint was incompatible; loaded %s", out_dir.name, weights_path.name)
            return model
        except (ValueError, OSError) as error:
            failures.append(f"{weights_path.name}: {error}")

    raise ValueError(f"could not load {cfg.model} weights from {out_dir}:\n" + "\n".join(failures))


load_trained_vit = load_trained_model   # backwards-compatible alias


def predictions_match_data(out_dir: str | Path, sd: SplitData, split: str = "test") -> bool:
    """True when ``pred_{split}.npz`` exists and its labels equal ``sd.y`` (same test set, same coding)."""
    p = Path(out_dir) / f"pred_{split}.npz"
    if not p.exists():
        return False
    y_saved = np.load(p)["y"]
    return len(y_saved) == len(sd.y) and np.array_equal(y_saved, sd.y)


def refresh_predictions(out_dir: str | Path, data: Dict[str, SplitData], splits=("val", "test")) -> Dict:
    """Re-predict a finished run from its saved weights and overwrite ``pred_*.npz`` / metrics in ``summary.json``.

    Use when the evaluation labels changed after training (e.g. a corrected ``test_label.npy``) - the
    trained weights stay valid, only the stored predictions/labels/metrics are regenerated.
    """
    out_dir = Path(out_dir)
    cfg = ExperimentConfig.load(out_dir / "config.json")
    model = load_trained_model(out_dir, data["train"].x.shape[1:])
    summary = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    for split in splits:
        sd = data[split]
        probs = predict_proba(model, sd.x)
        save_predictions(out_dir, split, probs, sd)
        summary[f"metrics_{split}"] = compute_metrics(sd.y, probs.argmax(1), probs)
    summary["repredicted"] = True
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("%s: predictions refreshed | test acc %.3f kappa %.3f", out_dir.name,
             summary["metrics_test"]["accuracy"], summary["metrics_test"]["kappa"])
    return summary


# ---------------------------------------------------------------------------
# Temporal-context experiment (Reviewer comment 1)
# ---------------------------------------------------------------------------
def embedding_fn(model):
    """Callable ``x -> (B, D)``: [CLS] embedding for the ViT, penultimate-layer features for functional models."""
    if hasattr(model, "embed"):
        return lambda xb: model.embed(xb, training=False)
    if getattr(model, "inputs", None) is not None and len(model.layers) >= 2:
        feat = tf.keras.Model(model.inputs, model.layers[-2].output)   # layer before the logits
        return lambda xb: feat(xb, training=False)
    return lambda xb: model(xb, training=False)


def extract_embeddings(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Epoch embeddings ``(N, D)`` used as input of the context models (see :func:`embedding_fn`)."""
    fn = embedding_fn(model)
    out = []
    for s in range(0, len(x), batch_size):
        out.append(fn(tf.constant(x[s:s + batch_size], tf.float32)).numpy())
    return np.concatenate(out, axis=0)


def _window_dataset(emb, y, subject, context, batch_size, shuffle, seed=0):
    win = make_context_windows(subject, context)
    e, mask = gather_windows(emb, win)
    yy, _ = gather_windows(y, win, pad_value=0)
    y1h = tf.keras.utils.to_categorical(yy, NUM_CLASSES).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(((e.astype(np.float32), mask), y1h, mask.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(min(len(e), 20_000), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), win


def run_context_experiment(embeddings: Dict[str, np.ndarray], data: Dict[str, SplitData], context: int = 10,
                           kind: str = "transformer", epochs: int = 30, lr: float = 3e-4, batch_size: int = 128,
                           out_dir: Optional[Path] = None, seed: int = 0, verbose: int = 1, **hp) -> Dict:
    """Train a context model on frozen epoch embeddings; evaluate the centre-epoch prediction.

    ``embeddings`` = ``{"train": (N,D), "val": ..., "test": ...}`` from :func:`extract_embeddings`.
    """
    set_seed(seed)
    for split in ("train", "test"):
        if data[split].seq is None:
            raise ValueError(f"{split} split has neither subject ids nor pseudo-recording ids - cannot build epoch sequences")
    L = 2 * context + 1
    model = build_context_model(kind, NUM_CLASSES, max_len=L, **hp)
    compile_model(model, lr, "adamw", 1e-4)

    if data["val"].seq is not None:
        tr_ds, _ = _window_dataset(embeddings["train"], data["train"].y, data["train"].seq, context, batch_size, True, seed)
        va_ds, _ = _window_dataset(embeddings["val"], data["val"].y, data["val"].seq, context, batch_size, False)
        eval_splits = ("val", "test")
    else:
        # validation split has no chronology (random epoch subset): hold out contiguous blocks of the
        # training sequences for early stopping instead
        from .data import block_split

        tr_idx, va_idx = block_split(len(data["train"]), 0.1, seed=seed, groups=data["train"].seq)
        y_tr, s_tr = data["train"].y, data["train"].seq
        tr_ds, _ = _window_dataset(embeddings["train"][tr_idx], y_tr[tr_idx], s_tr[tr_idx], context, batch_size, True, seed)
        va_ds, _ = _window_dataset(embeddings["train"][va_idx], y_tr[va_idx], s_tr[va_idx], context, batch_size, False)
        eval_splits = ("test",)
        log.info("context model: early stopping on contiguous blocks of the training sequences (val has no chronology)")
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(tr_ds, validation_data=va_ds, epochs=epochs, callbacks=cbs, verbose=verbose)

    result = {"context": context, "kind": kind, "n_params": count_params(model), "model_obj": model,
              "sequence_kind": data["test"].sequence_kind}
    for split in eval_splits:
        ds, win = _window_dataset(embeddings[split], data[split].y, data[split].seq, context, batch_size, False)
        logits = model.predict(ds, verbose=0)                              # (N, L, K)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[:, context]         # centre epoch
        result[f"probs_{split}"] = probs
        result[f"metrics_{split}"] = compute_metrics(data[split].y, probs.argmax(1), probs)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_predictions(out_dir, split, probs, data[split])
    return result
