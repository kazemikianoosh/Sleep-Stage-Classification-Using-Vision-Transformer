#!/usr/bin/env python
"""Run a trained sleep-stage model on local epochs without requiring labels.

The input must be a NumPy array shaped ``(epochs, samples, channels)`` or a
single epoch shaped ``(samples, channels)``. Patient data stay local: this
script only reads the file supplied with ``--input`` and optionally writes a
CSV containing class probabilities.

Examples
--------
python scripts/infer.py --run results/vit_ppg_rc_ra_s0 \
    --input /path/to/local_epochs.npy --input-channels PPG RC RA \
    --output predictions.csv

# Verify the inference path without patient data (predictions are meaningless):
python scripts/infer.py --run results/vit_ppg_rc_ra_s0 --demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import setup_logging  # also adds the repository root to sys.path

from vit_sleep import CHANNEL_NAMES, CLASS_NAMES
from vit_sleep.config import ExperimentConfig
from vit_sleep.data import channel_indices, preprocess_epochs


def _synthetic_epochs(n_epochs: int, n_samples: int, n_channels: int, seed: int) -> np.ndarray:
    """Create deterministic waveform-like input for a privacy-safe smoke test."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 30.0, n_samples, endpoint=False, dtype=np.float32)
    x = np.empty((n_epochs, n_samples, n_channels), dtype=np.float32)
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            frequency_hz = 0.2 + 0.15 * channel
            x[epoch, :, channel] = (
                np.sin(2 * np.pi * frequency_hz * t + 0.2 * epoch)
                + 0.05 * rng.standard_normal(n_samples)
            )
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path,
                        help="trained run containing config.json and best.weights.h5")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="local .npy array; it is never copied by this script")
    source.add_argument("--demo", action="store_true", help="use synthetic input to smoke-test inference")
    parser.add_argument("--input-channels", nargs="+", choices=CHANNEL_NAMES,
                        help="channel order in --input; inferred only for model-order or canonical five-channel arrays")
    parser.add_argument("--sample-size", type=int,
                        help="randomly select this many local epochs in memory; no sampled data file is written")
    parser.add_argument("--demo-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, help="optional destination CSV for predictions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()

    run_dir = args.run.expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        parser.error(f"{config_path} does not exist")
    cfg = ExperimentConfig.load(config_path)
    model_channels = list(cfg.data.channels)

    if args.demo:
        x = _synthetic_epochs(args.demo_epochs, cfg.data.target_len, len(model_channels), args.seed)
        available_channels = model_channels
        source_indices = np.arange(len(x))
        print("Using synthetic input; predicted stages are only an inference smoke test.")
    else:
        input_path = args.input.expanduser().resolve()
        if input_path.suffix.lower() != ".npy":
            parser.error("--input must be a .npy file")
        x = np.load(input_path, allow_pickle=False)
        if x.ndim == 2:
            x = x[None, ...]
        if x.ndim != 3:
            parser.error(f"expected (epochs, samples, channels), received shape {x.shape}")
        source_indices = np.arange(len(x))
        if args.sample_size is not None:
            if not 1 <= args.sample_size <= len(x):
                parser.error(f"--sample-size must be between 1 and {len(x)}")
            source_indices = np.sort(
                np.random.default_rng(args.seed).choice(len(x), args.sample_size, replace=False)
            )
            x = np.asarray(x[source_indices])
            print(f"Selected {len(x)} random local epochs; no sampled data file was written.")

        if args.input_channels:
            available_channels = list(args.input_channels)
        elif x.shape[-1] == len(model_channels):
            available_channels = model_channels
        elif x.shape[-1] == len(CHANNEL_NAMES):
            available_channels = list(CHANNEL_NAMES)
        else:
            parser.error("cannot infer channel order; provide --input-channels")

    if x.shape[-1] != len(available_channels):
        parser.error(
            f"array has {x.shape[-1]} channels but {len(available_channels)} names were provided"
        )

    try:
        x = x[..., channel_indices(model_channels, available_channels)]
    except KeyError as exc:
        parser.error(str(exc))
    x = preprocess_epochs(x, target_len=cfg.data.target_len, normalisation=cfg.data.normalisation)

    # Import TensorFlow-backed helpers only after input validation.
    from vit_sleep.train import configure_gpu, load_trained_model, predict_proba

    configure_gpu()
    model = load_trained_model(run_dir, input_shape=(cfg.data.target_len, len(model_channels)))
    probabilities = predict_proba(model, x, batch_size=args.batch_size)
    predicted = probabilities.argmax(axis=1)

    result = pd.DataFrame({
        "epoch": np.arange(len(predicted)),
        "source_index": source_indices,
        "predicted_index": predicted,
        "predicted_stage": [CLASS_NAMES[i] for i in predicted],
        **{f"prob_{name}": probabilities[:, i] for i, name in enumerate(CLASS_NAMES)},
    })
    print(result.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if len(result) > 10:
        print(f"... {len(result) - 10} additional epochs")

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
