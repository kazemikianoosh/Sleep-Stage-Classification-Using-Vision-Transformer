# ViT-Sleep

Minimal training and inference code for **An Interpretable Sleep Stage Classification Using Vision Transformer**.
The model classifies 30-s cardiorespiratory epochs into Wake, N1, N2, N3, and REM.

> **Privacy:** This repository contains no patient recordings, labels, identifiers,
> predictions, or trained checkpoints. `examples/synthetic_epochs.npy` is generated
> mathematically and is not derived from a participant.

## Contents

```text
vit_sleep/              model, preprocessing, training, metrics, interpretation
scripts/
  preprocess_edf.py     raw EDF/annotations to NumPy arrays
  train.py              train one model
  evaluate.py           evaluate saved predictions
  infer.py              run inference on local epochs
examples/               synthetic inference input
data/README.md          private-data format
tests/                  unit and model smoke tests
```

## Installation

```bash
git clone https://github.com/kazemikianoosh/Sleep-Stage-Classification-Using-Vision-Transformer.git
cd ViT_Sleep
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional preprocessing dependencies:

```bash
pip install -e ".[preprocess]"
```

## Data

Patient data must remain outside Git. Place authorized arrays under `data/`:

```text
train_data.npy          (N_train, samples, channels)
train_label.npy         (N_train,)
test_data.npy           (N_test, samples, channels)
test_label.npy          (N_test,)
```

Channel order is `PPG, RF, RC, RA, HRV`; labels are
`0=Wake, 1=N1, 2=N2, 3=N3, 4=REM`. See `data/README.md`.

## Train

```bash
python scripts/train.py \
  --model vit \
  --channels PPG RC RA \
  --patch-size 20 \
  --epochs 45 \
  --batch-size 64
```

Add `--mixed-precision` on a compatible GPU.

## Evaluate

```bash
python scripts/evaluate.py --run results/vit_ppg_rc_ra_s0
```

## Inference

On private local epochs:

```bash
python scripts/infer.py \
  --run results/vit_ppg_rc_ra_s0 \
  --input /private/path/epochs.npy \
  --input-channels PPG RC RA \
  --output predictions.csv
```

Privacy-safe smoke test:

```bash
python scripts/infer.py --run results/vit_ppg_rc_ra_s0 --demo
```

Inference requires a locally trained run or separately released checkpoint.

## License

MIT License. See `LICENSE`.
