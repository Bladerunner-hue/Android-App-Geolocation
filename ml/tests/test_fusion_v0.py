"""Production fusion_v0 contracts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def test_context12_shape_and_no_location():
    from datetime import datetime, timezone

    from ml.context12 import CONTEXT_DIM, context12, modality_mask

    v = context12(datetime.now(timezone.utc), 0, None, None, None)
    assert v.shape == (CONTEXT_DIM,)
    assert v.dtype == np.float32
    assert v[-1] == 0.0  # has_location
    m = modality_mask(True, False)
    assert list(m) == [1.0, 0.0, 1.0]


def test_fusion_forward_and_mask():
    import tensorflow as tf

    from ml.fusion_v0 import NUM_VIBES, PERCEPTUAL_DIM, build_fusion_v0, count_trainable_params

    model = build_fusion_v0()
    n_params = count_trainable_params(model)
    assert n_params < 400_000, n_params
    assert n_params > 100_000, n_params
    n = 3
    x = {
        "image_embedding": tf.ones([n, 576]),
        "audio_embedding": tf.ones([n, 1024]) * 3.0,
        "context": tf.zeros([n, 12]),
        "modality_mask": tf.constant([[1, 0, 1], [1, 0, 1], [1, 1, 1]], tf.float32),
    }
    o = model(x, training=False)
    assert o["vibe_probabilities"].shape == (n, NUM_VIBES)
    assert o["perceptual_embedding"].shape == (n, PERCEPTUAL_DIM)
    # first two rows: audio masked → identical if only audio differs
    x2 = dict(x)
    x2["audio_embedding"] = tf.ones([n, 1024]) * 9.0
    o2 = model(x2, training=False)
    assert np.allclose(
        o["vibe_logits"].numpy()[:2], o2["vibe_logits"].numpy()[:2], atol=1e-5
    )


def test_wiring_fixture_validate_and_train(tmp_path):
    from ml.make_wiring_fixture import main as make_fix
    import ml.make_wiring_fixture as m

    # write under tmp by patching cwd
    import os

    cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        assert make_fix() == 0
        man = ROOT / "ml/data_sample/fusion_wiring/manifest.json"
        from ml.validate_fusion_dataset import main as val_main
        import sys as _sys

        _sys.argv = ["validate", "--manifest", str(man)]
        assert val_main() == 0

        from ml.train_fusion_v0 import train
        import argparse

        args = argparse.Namespace(
            manifest=man,
            weights_out=tmp_path / "w.weights.h5",
            epochs=3,
            batch_size=16,
            learning_rate=1e-3,
            weight_decay=1e-4,
            modality_dropout=0.1,
            dropout=0.1,
            patience=3,
            seed=0,
            cpu=True,
        )
        assert train(args) == 0
        assert args.weights_out.is_file()
    finally:
        os.chdir(cwd)


def test_export_refuses_missing_weights(tmp_path):
    from ml.export_tflite import export

    with pytest.raises(FileNotFoundError):
        export(
            weights=tmp_path / "no.h5",
            parity_data=tmp_path / "p.npz",
            out=tmp_path / "o.tflite",
            savedmodel_dir=tmp_path / "sm",
        )


def test_sound_bootstrap_esc10_filter(tmp_path):
    # minimal fake esc50 csv
    meta = tmp_path / "meta"
    meta.mkdir()
    (tmp_path / "audio").mkdir()
    (meta / "esc50.csv").write_text(
        "filename,fold,target,category,esc10,src_file,take\n"
        "1-100032-A-0.wav,1,0,dog,True,x,A\n"
        "1-100038-A-14.wav,1,14,chirping_birds,False,y,A\n"
    )
    from ml.sound_bootstrap import load_esc10_rows

    rows = load_esc10_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["category"] == "dog"
    assert rows[0]["task"] == "sound_event"


def test_prepare_accepts_precomputed_embeddings(tmp_path):
    from datetime import datetime, timezone

    from ml.fusion_v0 import VIBE_LABELS

    lines = []
    for i in range(21):
        sid = f"s{i // 7}"  # 3 sessions
        lines.append(
            json.dumps(
                {
                    "sample_id": f"m{i}",
                    "session_id": sid,
                    "primary_vibe": VIBE_LABELS[i % 7],
                    "label_source": "human_self",
                    "consent_for_training": True,
                    "captured_at_utc": f"2026-0{(i % 9) + 1:d}-01T12:00:00Z".replace(
                        "2026-010", "2026-10"
                    )
                    if False
                    else datetime(2026, 1 + (i % 6), 1 + (i % 20), tzinfo=timezone.utc).isoformat(),
                    "utc_offset_minutes": 60,
                    "latitude": 48.0,
                    "longitude": 2.0,
                    "accuracy_m": 5.0,
                    "image_embedding": np.random.randn(576).tolist(),
                    "audio_embedding": np.random.randn(1024).tolist(),
                }
            )
        )
    jpath = tmp_path / "export.jsonl"
    jpath.write_text("\n".join(lines))
    out = tmp_path / "ds"
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.prepare_fusion_dataset",
            "--input",
            str(jpath),
            "--out-dir",
            str(out),
            "--skip-encoders",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "manifest.json").is_file()
