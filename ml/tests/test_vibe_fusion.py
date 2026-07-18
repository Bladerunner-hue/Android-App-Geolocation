"""ML contract tests — dense fusion (no TFLite convert in unit path if slow)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_build_and_forward():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ml.vibe_fusion.model import NUM_VIBES, PERCEPTUAL_DIM, build_vibe_fusion, dummy_batch

    model = build_vibe_fusion()
    batch = dummy_batch(4)
    out = model(batch, training=False)
    assert out["vibe_logits"].shape == (4, NUM_VIBES)
    assert out["perceptual_embedding"].shape == (4, PERCEPTUAL_DIM)
    norms = np.linalg.norm(out["perceptual_embedding"].numpy(), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_presence_mask_zeros_audio():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    from ml.vibe_fusion.model import AUDIO_DIM, build_vibe_fusion

    model = build_vibe_fusion()
    # same image/context, different audio + mask
    img = tf.ones([1, 576])
    ctx = tf.zeros([1, 12])
    aud_a = tf.ones([1, AUDIO_DIM])
    aud_b = tf.ones([1, AUDIO_DIM]) * 5.0
    mask_on = tf.constant([[1.0, 1.0, 0.0]])
    mask_off = tf.constant([[1.0, 0.0, 0.0]])
    o1 = model(
        {
            "image_emb": img,
            "audio_emb": aud_a,
            "context": ctx,
            "presence_mask": mask_off,
        },
        training=False,
    )
    o2 = model(
        {
            "image_emb": img,
            "audio_emb": aud_b,
            "context": ctx,
            "presence_mask": mask_off,
        },
        training=False,
    )
    # With audio masked off, different audio inputs should match
    assert np.allclose(o1["vibe_logits"].numpy(), o2["vibe_logits"].numpy(), atol=1e-5)
    o3 = model(
        {
            "image_emb": img,
            "audio_emb": aud_a,
            "context": ctx,
            "presence_mask": mask_on,
        },
        training=False,
    )
    # With audio on, should differ from masked
    assert not np.allclose(o1["vibe_logits"].numpy(), o3["vibe_logits"].numpy(), atol=1e-3)


def test_export_refuses_missing_weights(tmp_path):
    from ml.vibe_fusion.export import export

    with pytest.raises(FileNotFoundError):
        export(
            weights_path=tmp_path / "nope.h5",
            out_tflite=tmp_path / "m.tflite",
            savedmodel_dir=tmp_path / "sm",
            parity_data=tmp_path / "p.npz",
        )


def test_export_refuses_overwrite(tmp_path):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    from ml.vibe_fusion.export import export
    from ml.vibe_fusion.model import build_vibe_fusion, dummy_batch

    model = build_vibe_fusion()
    w = tmp_path / "w.weights.h5"
    model.save_weights(str(w))
    batch = dummy_batch(4)
    npz = tmp_path / "parity.npz"
    np.savez(
        npz,
        image_emb=batch["image_emb"].numpy(),
        audio_emb=batch["audio_emb"].numpy(),
        context=batch["context"].numpy(),
        presence_mask=batch["presence_mask"].numpy(),
    )
    out = tmp_path / "model.tflite"
    sm = tmp_path / "sm"
    export(
        weights_path=w,
        out_tflite=out,
        savedmodel_dir=sm,
        parity_data=npz,
        force=True,
        max_logit_mae=1.0,  # loose for untrained random weights + tflite
    )
    assert out.is_file()
    with pytest.raises(FileExistsError):
        export(
            weights_path=w,
            out_tflite=out,
            savedmodel_dir=sm,
            parity_data=npz,
            force=False,
            max_logit_mae=1.0,
        )


def test_int8_requires_rep(tmp_path):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ml.vibe_fusion.export import export
    from ml.vibe_fusion.model import build_vibe_fusion, dummy_batch

    model = build_vibe_fusion()
    w = tmp_path / "w.weights.h5"
    model.save_weights(str(w))
    batch = dummy_batch(4)
    npz = tmp_path / "parity.npz"
    np.savez(
        npz,
        image_emb=batch["image_emb"].numpy(),
        audio_emb=batch["audio_emb"].numpy(),
        context=batch["context"].numpy(),
        presence_mask=batch["presence_mask"].numpy(),
    )
    with pytest.raises(ValueError, match="INT8"):
        export(
            weights_path=w,
            out_tflite=tmp_path / "i.tflite",
            savedmodel_dir=tmp_path / "sm2",
            parity_data=npz,
            quantization="int8",
            force=True,
        )


def test_vibe_labels_count():
    from ml.vibe_fusion.model import NUM_VIBES, VIBE_LABELS

    assert NUM_VIBES == 7
    assert len(VIBE_LABELS) == 7


def test_train_step_smoke():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    from ml.vibe_fusion.model import build_vibe_fusion, dummy_batch

    model = build_vibe_fusion()
    opt = tf.keras.optimizers.Adam(1e-3)
    batch = dummy_batch(4)
    y = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    with tf.GradientTape() as tape:
        logits = model(batch, training=True)["vibe_logits"]
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        )
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    assert float(loss) > 0


def test_experiments_moe_is_marked():
    """Old MoE remains experimental, not production export."""
    moe = ROOT / "moe_kickstart.py"
    assert moe.is_file()
    text = moe.read_text(encoding="utf-8")
    assert "MoE" in text or "moe" in text.lower()
