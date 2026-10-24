#!/usr/bin/env python
"""Evaluate saved predictions: metrics with subject-level bootstrap CIs, confusion matrix, per-disorder table.

Example
-------
python scripts/evaluate.py --run results/vit_ppg_rc_ra_s0 --n-boot 1000
python scripts/evaluate.py --run results/vit_ppg_rc_ra_s0 --compare results/baselines/cnn_rnn_ppg_rc_ra_s0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--compare", default=None, help="second run for a paired bootstrap on kappa")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    setup_logging()

    from vit_sleep import CLASS_NAMES
    from vit_sleep.metrics import (bootstrap_ci, compute_metrics, confusion_df, format_ci, paired_bootstrap_delta,
                                   per_class_report, plot_confusion_matrix)
    from vit_sleep.train import load_predictions

    run = Path(args.run)
    pr = load_predictions(run, args.split)
    y, yp, probs, subj = pr["y"], pr["y_pred"], pr["probs"], pr["subject"]

    ci = bootstrap_ci(y, yp, subject=subj, n_boot=args.n_boot, probs=probs)
    ci.to_csv(run / f"metrics_{args.split}_ci.csv")
    print(f"\n== {run.name} ({args.split}; {'subject' if subj is not None else 'epoch'}-level bootstrap, n={args.n_boot}) ==")
    for k in ["accuracy", "f1_weighted", "f1_macro", "kappa"] + [f"f1_{c}" for c in CLASS_NAMES]:
        print(f"{k:14s} {format_ci(ci.loc[k])}")
    print("\n", per_class_report(y, yp).round(3).to_string())
    print("\nRow-normalised confusion matrix:\n", confusion_df(y, yp).round(3).to_string())

    # Table 2: per-disorder performance - from a test_disorder.npy array or from the legacy
    # per-disorder files (test_data_5calss_OSA.npy, ...), which are evaluated with the trained model.
    rows = {}
    if pr.get("disorder") is not None:
        for d in np.unique(pr["disorder"]):
            m = pr["disorder"] == d
            rows[str(d)] = compute_metrics(y[m], yp[m], probs[m])
    else:
        from vit_sleep.config import ExperimentConfig
        from vit_sleep.data import load_disorder_test_sets
        from vit_sleep.train import load_trained_model, predict_proba

        cfg = ExperimentConfig.load(run / "config.json")
        sets = load_disorder_test_sets(cfg.data.data_dir, cfg.data.channels, cfg.data.target_len, cfg.data.normalisation)
        if sets:
            model = load_trained_model(run, (cfg.data.target_len, len(cfg.data.channels)))
            for d, sd in sets.items():
                pb = predict_proba(model, sd.x)
                rows[d] = compute_metrics(sd.y, pb.argmax(1), pb)
    if rows:
        tab = pd.DataFrame(rows).T
        tab.to_csv(run / f"metrics_{args.split}_by_disorder.csv")
        print("\nBy disorder (Table 2):\n", tab[["accuracy", "kappa", "f1_weighted", "f1_N1"]].round(3).to_string())

    if args.compare:
        other = load_predictions(Path(args.compare), args.split)
        for key in ("kappa", "accuracy", "f1_N1"):
            d = paired_bootstrap_delta(y, other["y_pred"], yp, subject=subj, key=key, n_boot=args.n_boot)
            print(f"paired delta {key} ({run.name} - {Path(args.compare).name}): {d['delta']:+.3f} "
                  f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  p={d['p_two_sided']:.3f}")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = plot_confusion_matrix(y, yp, title=run.name)
        ax.figure.tight_layout()
        ax.figure.savefig(run / f"confusion_{args.split}.png", dpi=200)
        plt.close(ax.figure)


if __name__ == "__main__":
    main()
