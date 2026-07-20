"""Smoke tests for sparse MoE fusion — same I/O contract as fusion_v0."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def test_build_and_forward():
    from ml.experiments.fusion_moe_v0 import build_fusion_moe_v0
    from ml.fusion_v0 import dummy_batch

    m = build_fusion_moe_v0(num_experts=4, top_k=2)
    b = dummy_batch(4)
    o = m(b, training=True)
    assert o["vibe_probabilities"].shape == (4, 7)
    assert o["perceptual_embedding"].shape == (4, 128)
    # L2
    norms = tf.norm(o["perceptual_embedding"], axis=-1).numpy()
    assert np.allclose(norms, 1.0, atol=1e-4)
    # probs sum
    s = tf.reduce_sum(o["vibe_probabilities"], axis=-1).numpy()
    assert np.allclose(s, 1.0, atol=1e-4)


def test_same_input_names_as_dense():
    from ml.experiments.fusion_moe_v0 import build_fusion_moe_v0
    from ml.fusion_v0 import build_fusion_v0, dummy_batch

    d = build_fusion_v0()
    m = build_fusion_moe_v0()
    b = dummy_batch(2)
    od = d(b, training=False)
    om = m(b, training=False)
    for k in ("vibe_probabilities", "perceptual_embedding", "vibe_logits", "vibe_id"):
        assert k in od and k in om
        assert tuple(od[k].shape) == tuple(om[k].shape)


def test_training_step_runs():
    from ml.experiments.fusion_moe_v0 import build_fusion_moe_v0
    from ml.fusion_v0 import dummy_batch

    m = build_fusion_moe_v0(num_experts=4, top_k=1)
    opt = tf.keras.optimizers.Adam(1e-3)
    b = dummy_batch(4)
    y = tf.constant([0, 1, 2, 3], tf.int32)
    with tf.GradientTape() as tape:
        o = m(b, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y, o["vibe_logits"], from_logits=True
        )
        loss = tf.reduce_mean(loss)
        if m.losses:
            loss = loss + tf.add_n(m.losses)
    grads = tape.gradient(loss, m.trainable_variables)
    opt.apply_gradients(zip(grads, m.trainable_variables))
    assert float(loss) == float(loss)  # finite
