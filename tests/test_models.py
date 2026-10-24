"""Smoke tests for every model (skipped when TensorFlow is not installed)."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from vit_sleep import NUM_CLASSES  # noqa: E402
from vit_sleep.complexity import count_params, vit_flops  # noqa: E402
from vit_sleep.interpretability import (attention_rollout, attention_weighted_channel_importance,  # noqa: E402
                                        channel_token_contributions, cls_attention, gradient_channel_importance,
                                        head_projection_scores, occlusion_channel_importance)
from vit_sleep.models import (build_context_model, build_end_to_end, build_model, build_vit)  # noqa: E402
from vit_sleep.models.layers import AddPositionEmbedding  # noqa: E402

T, C = 400, 3


@pytest.fixture(scope="module")
def xy():
    rng = np.random.default_rng(0)
    x = rng.random((16, T, C), dtype=np.float32)
    y = rng.integers(0, NUM_CLASSES, 16)
    return x, y


@pytest.mark.parametrize("name", ["vit", "cnn_rnn", "ulw_sleepnet", "msa_cnn", "msa_cnn_small", "lite_transformer"])
def test_models_forward_and_train_step(name, xy):
    x, y = xy
    hp = {"patch_size": 20, "hidden_size": 32, "depth": 2, "num_heads": 2, "mlp_dim": 32} if name == "vit" else {}
    if name == "lite_transformer":
        hp = {"patch_size": 20}
    model = build_model(name, (T, C), NUM_CLASSES, **hp)
    out = model(x[:4], training=False)
    assert out.shape == (4, NUM_CLASSES)
    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    model.fit(x, tf.keras.utils.to_categorical(y, NUM_CLASSES), epochs=1, batch_size=8, verbose=0)
    assert count_params(model) > 0


def test_vit_introspection(xy):
    x, y = xy
    vit = build_vit((T, C), NUM_CLASSES, patch_size=20, hidden_size=32, depth=2, num_heads=2, mlp_dim=32)
    a = cls_attention(vit, x[:4])
    assert a.shape == (4, 2, T // 20) and np.allclose(a.sum(-1) <= 1.0 + 1e-5, True)
    r = attention_rollout(vit, x[:4])
    assert r.shape == (4, T // 20)
    ctb = channel_token_contributions(vit, x[:4])
    assert ctb.shape == (4, T // 20, C)
    imp = attention_weighted_channel_importance(vit, x[:4], channels=["PPG", "RC", "RA"])
    assert np.allclose(imp["per_epoch"].sum(1), 1.0, atol=1e-5)
    occ = occlusion_channel_importance(vit, x, y, ["PPG", "RC", "RA"])
    assert set(occ) == {"none", "PPG", "RC", "RA"}
    g = gradient_channel_importance(vit, x[:8], y[:8])
    assert g.shape == (C,) and abs(g.sum() - 1) < 1e-5
    hp = head_projection_scores(vit)
    assert hp["mean_minmax"].shape == (2,)
    f = vit_flops(T, C, 20, 32, 2, 2, 32)
    assert f["tokens"] == T // 20 + 1 and f["macs_total"] > 0


def test_vit_independent_patch_stride(xy):
    x, _ = xy
    kernel, stride = 20, 10
    expected_patches = (T - kernel) // stride + 1
    vit = build_vit((T, C), NUM_CLASSES, patch_size=kernel, patch_stride=stride,
                    hidden_size=32, depth=2, num_heads=2, mlp_dim=32)
    assert vit.embeddings.num_patches == expected_patches
    assert vit.embeddings(x[:2]).shape == (2, expected_patches + 1, 32)
    flops = vit_flops(T, C, kernel, 32, 2, 2, 32, patch_stride=stride)
    assert flops["tokens"] == expected_patches + 1


def test_fixed_position_embedding_accepts_float16():
    layer = AddPositionEmbedding(learnable=False)
    x = tf.ones((2, 10, 8), dtype=tf.float16)
    y = layer(x)
    assert y.dtype == tf.float16 and y.shape == x.shape


def test_weights_roundtrip(tmp_path, xy):
    x, _ = xy
    vit = build_vit((T, C), NUM_CLASSES, patch_size=20, hidden_size=32, depth=2, num_heads=2, mlp_dim=32)
    p = tmp_path / "w.weights.h5"
    vit.save_weights(str(p))
    vit2 = build_vit((T, C), NUM_CLASSES, patch_size=20, hidden_size=32, depth=2, num_heads=2, mlp_dim=32)
    vit2.load_weights(str(p))
    assert np.allclose(vit(x[:2], training=False), vit2(x[:2], training=False))


@pytest.mark.parametrize("kind", ["transformer", "bigru"])
def test_context_models(kind, xy):
    x, y = xy
    L, D = 5, 16
    emb = np.random.default_rng(0).random((8, L, D), dtype=np.float32)
    mask = np.ones((8, L), bool)
    mask[:, 0] = False
    ctx = build_context_model(kind, NUM_CLASSES, max_len=L, **({"d_model": 16, "depth": 1, "num_heads": 2, "mlp_dim": 16} if kind == "transformer" else {"units": 8}))
    out = ctx((emb, mask), training=False)
    assert out.shape == (8, L, NUM_CLASSES)
    ctx.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    yy = tf.keras.utils.to_categorical(np.random.default_rng(1).integers(0, 5, (8, L)), NUM_CLASSES)
    ctx.fit((emb, mask), yy, sample_weight=mask.astype(np.float32), epochs=1, verbose=0)

    vit = build_vit((T, C), NUM_CLASSES, patch_size=20, hidden_size=D, depth=1, num_heads=2, mlp_dim=16)
    e2e = build_end_to_end(vit, ctx, seq_len=L, input_shape=(T, C), freeze_encoder=True)
    xs = np.random.default_rng(2).random((2, L, T, C), dtype=np.float32)
    assert e2e((xs, np.ones((2, L), bool)), training=False).shape == (2, L, NUM_CLASSES)
