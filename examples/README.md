# Synthetic inference sample

`synthetic_epochs.npy` contains four generated 30-s epochs with channels `PPG, RC, RA` and shape `(4, 2000, 3)`. It contains no participant data and its model predictions have no clinical meaning.

```bash
python scripts/infer.py --run /path/to/local/trained/run --input examples/synthetic_epochs.npy --input-channels PPG RC RA
```
