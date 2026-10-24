#!/usr/bin/env python
"""Train and evaluate one epoch-level model.

Examples
--------
# ViT-1D on PPG + chest + abdominal effort (best configuration of the paper)
python scripts/train.py --model vit --channels PPG RC RA --patch-size 20 --epochs 45

# PPG-only ViT with class re-weighting to probe the N1 trade-off (Reviewer 3)
python scripts/train.py --model vit --channels PPG --class-weight sqrt --name vit_ppg_sqrtcw

# Baselines under the identical inter-patient protocol (Reviewer 4)
python scripts/train.py --model cnn_rnn --channels PPG RC RA
python scripts/train.py --model ulw_sleepnet --channels PPG RC RA
python scripts/train.py --model msa_cnn --channels PPG RC RA
python scripts/train.py --model lite_transformer --channels PPG RC RA
"""
from __future__ import annotations

import argparse

from _common import add_data_args, add_train_args, add_vit_args, config_from_args, setup_logging


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit",
                   choices=["vit", "cnn_rnn", "ulw_sleepnet", "msa_cnn", "msa_cnn_small", "lite_transformer"])
    p.add_argument("--name", default=None, help="experiment name (results/<name>/)")
    add_data_args(p)
    add_train_args(p)
    add_vit_args(p)
    args = p.parse_args()
    setup_logging(not args.quiet)

    from vit_sleep.train import configure_gpu, run_experiment

    configure_gpu(mixed_precision=args.mixed_precision)
    cfg = config_from_args(args, model=args.model, name=args.name)
    summary = run_experiment(cfg, verbose=0 if args.quiet else 1, skip_existing=not args.force)
    m = summary["metrics_test"]
    tag = "loaded existing run" if summary.get("skipped") else "test"
    print(f"\n[{cfg.name}] {tag}  acc={m['accuracy']:.3f}  F1w={m['f1_weighted']:.3f}  kappa={m['kappa']:.3f}  "
          f"F1/class=({m['f1_Wake']:.2f}, {m['f1_N1']:.2f}, {m['f1_N2']:.2f}, {m['f1_N3']:.2f}, {m['f1_REM']:.2f})  "
          f"params={summary['n_params']:,}")


if __name__ == "__main__":
    main()
