#!/usr/bin/env python
"""Raw PSG (EDF + hypnogram CSV) -> ``data/{train,val,test}_{data,label,subject}.npy`` (Fig. 1 of the paper).

Directory layout expected::

    <raw_dir>/
        <subject_id>.edf
        <subject_id>.csv            # hypnogram: columns 'epoch' (or 'onset' in s) and 'stage' (W,N1,N2,N3,R)
        [disorders.csv]             # optional: subject_id,disorder

Natus channel labels vary between exports - pass them explicitly.

Example
-------
python scripts/preprocess_edf.py --raw-dir /path/to/psg --out-dir data \
    --ppg Pleth --flow "Nasal Pressure" --chest Chest --abdomen Abdomen --seed 0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _common import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--ppg", default="Pleth")
    p.add_argument("--flow", default="Flow")
    p.add_argument("--chest", default="Chest")
    p.add_argument("--abdomen", default="Abdomen")
    p.add_argument("--target-fs", type=int, default=64)
    p.add_argument("--downsample", default="linear", choices=["linear", "decimate"])
    p.add_argument("--fractions", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    setup_logging()
    log = logging.getLogger("preprocess")

    from vit_sleep.preprocessing import (ChannelMap, PreprocessConfig, process_recording, save_splits,
                                         split_subjects)

    cfg = PreprocessConfig(target_fs=args.target_fs, downsample_method=args.downsample,
                           channel_map=ChannelMap(args.ppg, args.flow, args.chest, args.abdomen))
    raw = Path(args.raw_dir)
    edfs = sorted(raw.glob("*.edf")) + sorted(raw.glob("*.EDF"))
    if not edfs:
        raise SystemExit(f"no EDF files in {raw}")

    xs, ys, ss = [], [], []
    for edf in edfs:
        hyp = edf.with_suffix(".csv")
        if not hyp.exists():
            log.warning("no hypnogram for %s - skipped", edf.name)
            continue
        try:
            x, y, s = process_recording(edf, hyp, cfg, subject_id=edf.stem)
        except Exception as e:  # keep going, report at the end
            log.error("%s failed: %s", edf.name, e)
            continue
        xs.append(x), ys.append(y), ss.append(s)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    subject = np.concatenate(ss)
    disorder = None
    dis_file = raw / "disorders.csv"
    if dis_file.exists():
        d = pd.read_csv(dis_file).set_index("subject_id")["disorder"].to_dict()
        disorder = np.array([d.get(s, "unknown") for s in subject])

    split_map = split_subjects(subject, tuple(args.fractions), args.seed)
    save_splits(args.out_dir, x, y, subject, split_map, disorder)
    pd.Series({k: v for k, v in split_map.items()}).to_json(Path(args.out_dir) / "split_subjects.json", indent=2)
    log.info("done: %d epochs, %d subjects, class counts %s", len(y), len(np.unique(subject)), np.bincount(y, minlength=5))


if __name__ == "__main__":
    main()
