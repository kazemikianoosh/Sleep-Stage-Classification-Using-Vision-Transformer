"""Experiment configuration.

Two presets are provided:

* :func:`code_defaults` – the hyper-parameters that were actually used by the
  notebooks that produced the submitted results (hidden 256, MLP 128, patch 20,
  stochastic depth 0.8, AdamW 1e-4, 45 epochs, per-epoch min-max normalisation of
  signals resampled to 2000 samples).
* :func:`paper_defaults` – the hyper-parameters as *written* in Section 2.4 of the
  manuscript (768 filters, MLP 256, survival probability 0.9, two 1024-unit head
  layers, LR drop after 15 epochs, early stopping).

See ``docs/revision_checklist.md`` for the discrepancies that must be reconciled
in the revised manuscript.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class DataConfig:
    data_dir: str = "data"
    channels: List[str] = field(default_factory=lambda: ["PPG", "RC", "RA"])
    target_len: int = 2000              # samples per epoch after resampling (2000 = legacy; 1920 = 64 Hz exact)
    normalisation: str = "minmax"       # "minmax" (per epoch, per channel) | "zscore" | "none"
    val_fraction: float = 0.15          # used only if no explicit validation split is on disk
    batch_size: int = 64


@dataclass
class ViTConfig:
    patch_size: int = 20
    patch_stride: Optional[int] = None       # None -> patch_size (original non-overlapping patches)
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 6
    mlp_dim: int = 128
    dropout: float = 0.0
    attention_dropout: float = 0.0
    sd_survival_probability: float = 0.8
    head_hidden_units: Tuple[int, ...] = ()   # () = single linear head (code); (1024, 1024) = paper text


@dataclass
class TrainConfig:
    epochs: int = 45
    learning_rate: float = 1e-4
    lr_drop_epoch: Optional[int] = None    # e.g. 15 -> LR /10 after epoch 15 (paper text)
    weight_decay: float = 1e-4
    optimizer: str = "adamw"               # "adamw" | "adam"
    early_stopping_patience: Optional[int] = None
    class_weight: Optional[str] = None     # None | "balanced" | "sqrt"
    label_smoothing: float = 0.0
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "vit_ppg_rc_ra"
    model: str = "vit"                     # vit | cnn_rnn | ulw_sleepnet | msa_cnn | lite_transformer
    data: DataConfig = field(default_factory=DataConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "results"

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        d["data"] = DataConfig(**d.get("data", {}))
        vit = dict(d.get("vit", {}))
        if "head_hidden_units" in vit:
            vit["head_hidden_units"] = tuple(vit["head_hidden_units"])
        d["vit"] = ViTConfig(**vit)
        d["train"] = TrainConfig(**d.get("train", {}))
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def code_defaults(channels: Sequence[str] = ("PPG", "RC", "RA")) -> ExperimentConfig:
    """Hyper-parameters used by the original notebooks."""
    cfg = ExperimentConfig()
    cfg.data.channels = list(channels)
    cfg.name = "vit_" + "_".join(c.lower() for c in channels)
    return cfg


def paper_defaults(channels: Sequence[str] = ("PPG", "RC", "RA")) -> ExperimentConfig:
    """Hyper-parameters as written in Section 2.4 of the manuscript."""
    cfg = code_defaults(channels)
    cfg.vit = ViTConfig(
        patch_size=20, hidden_size=768, depth=6, num_heads=6, mlp_dim=256,
        dropout=0.1, attention_dropout=0.1, sd_survival_probability=0.9,
        head_hidden_units=(1024, 1024),
    )
    cfg.train = TrainConfig(
        epochs=45, learning_rate=1e-4, lr_drop_epoch=15, weight_decay=1e-4,
        optimizer="adamw", early_stopping_patience=8,
    )
    cfg.name += "_paper"
    return cfg
