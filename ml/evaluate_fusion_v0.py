"""Held-out evaluation for fusion_v0 (macro-F1, per-class, NLL, confusion)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow as tf

    from ml.fusion_v0 import VIBE_LABELS, build_fusion_v0
    from ml.train_fusion_v0 import evaluate, load_npz

    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True, help="Held-out test NPZ")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    data = load_npz(args.npz)
    model = build_fusion_v0()
    model.load_weights(str(args.weights))
    summary = evaluate(model, data)

    x = {
        "image_embedding": tf.constant(data["image_embedding"]),
        "audio_embedding": tf.constant(data["audio_embedding"]),
        "context": tf.constant(data["context"]),
        "modality_mask": tf.constant(data["modality_mask"]),
    }
    probs = model(x, training=False)["vibe_probabilities"].numpy()
    y = data["vibe_label"]
    pred = probs.argmax(-1)
    cm = np.zeros((len(VIBE_LABELS), len(VIBE_LABELS)), dtype=np.int64)
    for t, pr in zip(y, pred):
        cm[int(t), int(pr)] += 1
    per_class = {}
    for i, name in enumerate(VIBE_LABELS):
        tp = cm[i, i]
        support = cm[i].sum()
        pred_c = cm[:, i].sum()
        prec = float(tp / (pred_c + 1e-9))
        rec = float(tp / (support + 1e-9))
        per_class[name] = {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec + 1e-9),
            "support": int(support),
        }
    report = {
        **summary,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(VIBE_LABELS),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
