# Data directory

The PSG recordings used in the paper (UCI Sleep Center, IRB #267) contain
protected health information and **cannot be redistributed**. This directory is
git-ignored; place your own exported arrays here.

## Expected files

```
data/
├── train_data.npy       float32  (N_train, T, 5)
├── train_label.npy      int      (N_train,)          0=Wake 1=N1 2=N2 3=N3 4=REM
├── train_subject.npy    int/str  (N_train,)          subject id of every epoch (optional but required for
│                                                        temporal-context models, HMM smoothing, subject-level CIs)
├── val_data.npy / val_label.npy / val_subject.npy      (optional; otherwise a subject-wise split of train is used)
├── test_data.npy / test_label.npy / test_subject.npy
└── test_disorder.npy    str      (N_test,)           e.g. "OSA", "Insomnia", "Hypersomnia" (optional, for Table 2)
```

* `T` is the number of samples per 30-s epoch (1920 at 64 Hz; the historical export had 1921).
  `vit_sleep.data.preprocess_epochs` resamples to `target_len` (default 2000) at load time.
* Channel order is fixed: `["PPG", "RF", "RC", "RA", "HRV"]`
  (finger PPG, respiratory flow, chest effort, abdominal effort, PPG-derived HRV tachogram).
* Epochs **must be stored in chronological order within each subject**; the sequence models
  reconstruct hypnograms from `(subject, order-in-array)`.

## Legacy layout (the arrays exported for the submitted paper)

The original notebooks read these files from the repository root, and that layout is fully supported —
no renaming needed. Files are searched in `data/` first and then in the repository root:

```
ViT_Sleep/                          (or ViT_Sleep/data/)
├── train_data_5calss.npy           (N_train, 1921, 5)     -> resampled to 2000 samples at load time
├── train_label_5calss.npy
├── test_data_5calss.npy
├── test_label_5calss.npy
├── test_data_5calss_OSA.npy          test_label_5calss_OSA.npy          }  per-disorder test sets
├── test_data_5calss_Insomnia.npy     test_label_5calss_Insomnia.npy     }  (Table 2); also used for
└── test_data_5calss_Hypersomnia.npy  test_label_5calss_Hypersomnia.npy  }  train_*_Hypersomnia.npy etc.
```

`*.npy` is git-ignored, so these patient-data files stay local even when placed at the root.

## Segment-level mode (no `*_subject.npy`)

The train/test split on disk is used as it is (it was made subject-wise when the arrays were exported).
Without subject ids the code automatically switches to *segment level*:

* the validation set is a reproducible random 15 % of the training epochs (`val_split="random"`, the
  setting of the reference runs) or contiguous blocks (`val_split="blocks"`, stricter, but changes the
  validation set); bootstrap CIs are epoch-level;
* **pseudo-recordings**: if the stored epoch order is chronological (self-transition rate ≫ shuffled
  baseline — checked automatically), the order is used as the hypnogram. When the per-disorder files are
  contiguous slices of the full arrays, each slice becomes one pseudo-recording and every epoch gets a
  disorder label (Table 2, per-disorder analyses). This enables §3 of the notebook and
  `scripts/run_sequence.py` at segment level. Caveat to state in the paper: subject boundaries inside a
  pseudo-recording are unknown (≤ n_subjects − 1 spurious transitions per split; context windows may
  straddle two subjects there); HMM hyper-parameters stay at defaults because the random validation
  subset has no chronology.

## Label coding check

`load_experiment_data` refuses to run when train and test label files use different codings (e.g. a
test file with N1/N2 merged: classes 0–3 only, ~46 % in class 1). All arrays must use
`0 Wake, 1 N1, 2 N2, 3 N3, 4 REM`. `vit_sleep.data.inspect_label_files(DATA_DIR)` lists every
`*label*.npy` with its class percentages — delete or rename the inconsistent file so the loader falls
back to the correct one (search order: `test_label.npy`, then `test_label_5calss.npy`, `test_label_5class.npy`).

To regenerate these arrays from raw EDF + annotation files, see `scripts/preprocess_edf.py`.
