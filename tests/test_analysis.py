"""Unit tests for the NumPy-only analysis modules (no TensorFlow required)."""
import numpy as np
import pandas as pd
import pytest

from vit_sleep import CLASS_NAMES, NUM_CLASSES
from vit_sleep.data import (channel_indices, gather_windows, make_context_windows, normalise_epochs,
                            resample_epochs, subject_split)
from vit_sleep.features import epoch_features, pairwise_feature_auc
from vit_sleep.metrics import bootstrap_ci, compute_metrics, confusion_df, entropy_confidence
from vit_sleep.transitions import (bout_lengths, context_breakdown, hmm_smooth, n1_by_previous_stage,
                                   previous_stage, transition_mask, transition_matrix, viterbi_decode)


def _hypnogram(n_subjects=3, n_epochs=200, seed=0):
    rng = np.random.default_rng(seed)
    P = np.array([[.9, .08, .02, 0, 0], [.1, .6, .3, 0, 0], [.02, .05, .85, .05, .03],
                  [0, 0, .1, .9, 0], [.03, .02, .1, 0, .85]])
    ys, subs = [], []
    for s in range(n_subjects):
        y = [0]
        for _ in range(n_epochs - 1):
            y.append(rng.choice(5, p=P[y[-1]]))
        ys.append(np.array(y)), subs.append(np.full(n_epochs, s))
    return np.concatenate(ys), np.concatenate(subs), P


def test_metrics_have_no_offsets():
    y = np.array([0, 1, 2, 3, 4] * 20)
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0 and m["kappa"] == 1.0 and m["f1_N1"] == 1.0
    yp = y.copy()
    yp[::2] = (yp[::2] + 1) % 5
    m = compute_metrics(y, yp)
    assert abs(m["accuracy"] - 0.5) < 1e-9
    cm = confusion_df(y, y)
    assert np.allclose(np.diag(cm.to_numpy()), 1.0)


def test_entropy_confidence_bounds():
    probs = np.array([[1, 0, 0, 0, 0], [.2] * 5])
    c = entropy_confidence(probs)
    assert c[0] > 0.999 and abs(c[1]) < 1e-6


def test_bootstrap_ci_subject_level():
    y, s, _ = _hypnogram()
    rng = np.random.default_rng(1)
    yp = np.where(rng.random(len(y)) < 0.8, y, rng.integers(0, 5, len(y)))
    ci = bootstrap_ci(y, yp, subject=s, n_boot=50)
    assert (ci["ci_low"] <= ci["value"] + 1e-9).all() and (ci["value"] <= ci["ci_high"] + 1e-9).all()


def test_transition_matrix_and_viterbi_improve_noisy_posteriors():
    y, s, P_true = _hypnogram(n_subjects=4, n_epochs=400)
    P = transition_matrix(y, s)
    assert np.allclose(P.sum(1), 1.0)
    assert P[3, 3] > 0.5 and P[0, 0] > 0.5
    rng = np.random.default_rng(2)
    probs = np.full((len(y), 5), 0.1)
    probs[np.arange(len(y)), y] = 0.6
    flip = rng.random(len(y)) < 0.35
    probs[flip] = np.roll(probs[flip], 1, axis=1)          # noisy posteriors
    probs /= probs.sum(1, keepdims=True)
    base = compute_metrics(y, probs.argmax(1))["accuracy"]
    smoothed = hmm_smooth(probs, s, P, np.bincount(y, minlength=5) / len(y))
    assert compute_metrics(y, smoothed)["accuracy"] > base
    post = hmm_smooth(probs, s, P, mode="posterior")
    assert post.shape == probs.shape and np.allclose(post.sum(1), 1.0)


def test_viterbi_recovers_deterministic_path():
    log_em = np.log(np.array([[.9, .1], [.1, .9], [.1, .9]]))
    log_tr = np.log(np.array([[.5, .5], [.5, .5]]))
    assert viterbi_decode(log_em, log_tr, np.log([.5, .5])).tolist() == [0, 1, 1]


def test_transition_mask_and_previous_stage():
    y = np.array([0, 0, 1, 2, 2, 2, 1, 0])
    s = np.zeros(8, dtype=int)
    m = transition_mask(y, s, radius=1)
    assert m.tolist() == [False, True, True, True, False, True, True, True]
    prev = previous_stage(y, s)
    assert prev.tolist() == [-1, -1, 0, 1, 1, 1, 2, 1]
    df = n1_by_previous_stage(y, y, s)
    assert set(df.index) <= {"Wake", "N2", "start", "N1", "N3", "REM"}
    b = bout_lengths(y, s)
    assert b["length"].sum() == 8
    cb = context_breakdown(y, y, s)
    assert cb.loc["all", "accuracy"] == 1.0


def test_context_windows_do_not_cross_subjects():
    s = np.array([0, 0, 0, 1, 1])
    w = make_context_windows(s, context=1)
    assert w.tolist() == [[-1, 0, 1], [0, 1, 2], [1, 2, -1], [-1, 3, 4], [3, 4, -1]]
    x = np.arange(5)[:, None].astype(float)
    g, mask = gather_windows(x, w)
    assert g.shape == (5, 3, 1) and mask.sum() == 11 and g[0, 0, 0] == 0.0


def test_resample_and_normalise():
    x = np.random.default_rng(0).random((5, 1921, 5)).astype(np.float32)
    r = resample_epochs(x, 2000)
    assert r.shape == (5, 2000, 5)
    t = resample_epochs(x, 1920, method="truncate")
    assert t.shape == (5, 1920, 5) and np.allclose(t, x[:, :1920])
    n = normalise_epochs(r, "minmax")
    assert n.min() >= 0 and n.max() <= 1 + 1e-6
    z = normalise_epochs(r, "zscore")
    assert np.allclose(z.mean(1), 0, atol=1e-4)
    assert channel_indices(["PPG", "RC", "RA"]) == [0, 2, 3]


def test_pseudo_sequences_from_legacy_layout(tmp_path):
    from vit_sleep.data import (block_split, check_label_consistency, chronology_score, infer_pseudo_sequences,
                                locate_subsequence, load_experiment_data)

    y, s, _ = _hypnogram(n_subjects=6, n_epochs=150)
    assert chronology_score(y) > 0.6
    assert chronology_score(np.random.default_rng(0).permutation(y)) < 0.5
    # legacy layout: full arrays + per-disorder contiguous slices, no subject ids
    T = 40
    x = np.random.default_rng(0).random((len(y), T, 5), dtype=np.float32)
    np.save(tmp_path / "train_data_5calss.npy", x); np.save(tmp_path / "train_label_5calss.npy", y)
    np.save(tmp_path / "test_data_5calss.npy", x[:300]); np.save(tmp_path / "test_label_5calss.npy", y[:300])
    slices = {"OSA": (0, 300), "Insomnia": (300, 600), "Hypersomnia": (600, 900)}
    for d, (a, b) in slices.items():
        np.save(tmp_path / f"train_label_5calss_{d}.npy", y[a:b]); np.save(tmp_path / f"train_data_5calss_{d}.npy", x[a:b])
    assert locate_subsequence(y, y[300:600]) == 300 and locate_subsequence(y, np.array([9, 9, 9])) is None
    seq, dis, info = infer_pseudo_sequences(tmp_path, "train", y)
    assert info["accepted"] and set(np.unique(seq)) == {1, 2, 3} and (dis[0] == "OSA") and (dis[899] == "Hypersomnia")
    data = load_experiment_data(tmp_path, ["PPG", "RC"], target_len=T, pseudo_sequences="auto")
    assert data["train"].sequence_kind == "pseudo-gapped" and data["train_hypnogram"].seq is not None
    assert data["val"].seq is None and data["test"].seq is not None and data["test"].sequence_kind == "pseudo"
    assert data["train_hypnogram"].x.shape[0] == 0 and data["train_hypnogram"].x.shape[-1] == 2
    tr, va = block_split(900, 0.15, groups=seq)
    assert len(set(tr) & set(va)) == 0 and 0.05 < len(va) / 900 < 0.3
    # inconsistent label coding is refused
    bad = y[:300].copy(); bad[bad == 1] = 2; bad[bad == 4] = 3
    np.save(tmp_path / "test_label_5calss.npy", bad)
    assert not check_label_consistency(y, bad)["ok"]
    with pytest.raises(ValueError):
        load_experiment_data(tmp_path, ["PPG"], target_len=T)


def test_subject_split_is_disjoint():
    s = np.repeat(np.arange(20), 10)
    tr, va = subject_split(s, 0.2, seed=0)
    assert not set(s[tr]) & set(s[va]) and len(set(s[va])) == 4


def test_epoch_features_on_synthetic_signals():
    fs, T = 2000 / 30, 2000
    t = np.arange(T) / fs
    rng = np.random.default_rng(0)
    n = 40
    hr = np.where(np.arange(n) < 20, 60, 80)          # two "stages" with different heart rate
    x = np.zeros((n, T, 3), np.float32)
    for i in range(n):
        x[i, :, 0] = np.sin(2 * np.pi * hr[i] / 60 * t) + 0.05 * rng.standard_normal(T)
        x[i, :, 1] = np.sin(2 * np.pi * 0.25 * t) + 0.05 * rng.standard_normal(T)
        x[i, :, 2] = np.sin(2 * np.pi * 0.25 * t + 0.3) + 0.05 * rng.standard_normal(T)
    f = epoch_features(x, ["PPG", "RC", "RA"])
    assert abs(f["hr_mean"].iloc[0] - 60) < 3 and abs(f["hr_mean"].iloc[-1] - 80) < 3
    assert abs(f["rc_rate"].iloc[0] - 15) < 2
    y = (np.arange(n) >= 20).astype(int)
    auc = pairwise_feature_auc(f, y, pairs=((1, 0),))
    assert auc.loc["hr_mean"].iloc[0] > 0.95
