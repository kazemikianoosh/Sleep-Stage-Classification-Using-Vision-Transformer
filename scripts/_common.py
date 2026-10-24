"""Shared CLI helpers (adds the repository root to ``sys.path`` so ``vit_sleep`` imports without installation)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vit_sleep import CHANNEL_NAMES  # noqa: E402

CHANNEL_CONFIGS = {
    # name -> channels ; the first five reproduce Table 1, the last three probe single-belt setups (Reviewer 2)
    "ppg": ["PPG"],
    "ppg_rc_ra": ["PPG", "RC", "RA"],
    "ppg_rf_rc_ra": ["PPG", "RF", "RC", "RA"],
    "ppg_rc_ra_hrv": ["PPG", "RC", "RA", "HRV"],
    "all5": ["PPG", "RF", "RC", "RA", "HRV"],
    "ppg_rc": ["PPG", "RC"],
    "ppg_ra": ["PPG", "RA"],
    "ppg_hrv": ["PPG", "HRV"],
}


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default=None,
                   help="folder with the *.npy arrays (default: data/ if populated, else the repository root "
                        "where the legacy train_data_5calss.npy files live)")
    p.add_argument("--channels", nargs="+", default=["PPG", "RC", "RA"], choices=CHANNEL_NAMES,
                   help="input channels in this order")
    p.add_argument("--target-len", type=int, default=2000, help="samples per epoch after resampling")
    p.add_argument("--normalisation", default="minmax", choices=["minmax", "zscore", "none"])
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", default=str(ROOT / "results"),
                   help="results root; every run is written to <output-dir>/<run-name>/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="retrain even if <output-dir>/<run-name>/pred_test.npz already exists (default: skip)")
    p.add_argument("--quiet", action="store_true")


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-drop-epoch", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    p.add_argument("--early-stopping", type=int, default=None, help="patience in epochs")
    p.add_argument("--class-weight", default=None, choices=[None, "balanced", "sqrt"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mixed-precision", action="store_true",
                   help="train in mixed_float16 on GPU (1.5-3x faster on V100/A100; logits stay float32)")


def add_vit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--patch-size", type=int, default=20)
    p.add_argument("--patch-stride", type=int, default=None,
                   help="Conv1D patch stride (default: patch size, i.e. non-overlapping patches)")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--mlp-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--attention-dropout", type=float, default=0.0)
    p.add_argument("--sd-survival", type=float, default=0.8)
    p.add_argument("--head-hidden", type=int, nargs="*", default=[])


def resolve_args_data_dir(args) -> str:
    """Fill ``args.data_dir`` when not given: data/ if populated, otherwise the repository root."""
    from vit_sleep.data import resolve_data_dir

    if getattr(args, "data_dir", None) is None:
        args.data_dir = str(resolve_data_dir(ROOT))
    return args.data_dir


def config_from_args(args, model: str = "vit", name: str | None = None):
    from vit_sleep.config import DataConfig, ExperimentConfig, TrainConfig, ViTConfig

    resolve_args_data_dir(args)
    cfg = ExperimentConfig(
        name=name or f"{model}_" + "_".join(c.lower() for c in args.channels) + f"_s{args.seed}",
        model=model,
        data=DataConfig(data_dir=args.data_dir, channels=list(args.channels), target_len=args.target_len,
                        normalisation=args.normalisation, val_fraction=args.val_fraction, batch_size=args.batch_size),
        vit=ViTConfig(patch_size=getattr(args, "patch_size", 20),
                      patch_stride=getattr(args, "patch_stride", None),
                      hidden_size=getattr(args, "hidden_size", 256),
                      depth=getattr(args, "depth", 6), num_heads=getattr(args, "num_heads", 6),
                      mlp_dim=getattr(args, "mlp_dim", 128), dropout=getattr(args, "dropout", 0.0),
                      attention_dropout=getattr(args, "attention_dropout", 0.0),
                      sd_survival_probability=getattr(args, "sd_survival", 0.8),
                      head_hidden_units=tuple(getattr(args, "head_hidden", []) or [])),
        train=TrainConfig(epochs=args.epochs, learning_rate=args.lr, lr_drop_epoch=args.lr_drop_epoch,
                          weight_decay=args.weight_decay, optimizer=args.optimizer,
                          early_stopping_patience=args.early_stopping, class_weight=args.class_weight,
                          label_smoothing=args.label_smoothing, seed=args.seed),
        output_dir=args.output_dir,
    )
    return cfg
