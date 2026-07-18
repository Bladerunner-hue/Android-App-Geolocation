"""
Train fusion_v0 dense head on precomputed NPZ features (human vibe labels only).

Do not use ml/train.py or synthetic MoE data for release models.

Example:
  python -m ml.train_fusion_v0 \\
    --manifest /secure/geojournal/fusion-v0/manifest.json \\
    --weights-out /secure/artifacts/fusion-v0-seed42.weights.h5 \\
    --epochs 60 --batch-size 32 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _setup_gpu(force_cpu: bool) -> None:
    import os

    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path)
    data = {k: z[k] for k in z.files}
    n = data["vibe_label"].shape[0]
    if "sample_weight" not in data:
        data["sample_weight"] = np.ones((n,), dtype=np.float32)
    return data


def make_dataset(
    data: Dict[str, np.ndarray],
    batch_size: int,
    training: bool,
    seed: int,
    modality_dropout_rate: float,
):
    import tensorflow as tf
    from ml.fusion_v0 import modality_dropout

    n = data["vibe_label"].shape[0]
    if n == 0:
        # empty placeholder
        spec = {
            "image_embedding": tf.TensorSpec([None, 576], tf.float32),
            "audio_embedding": tf.TensorSpec([None, 1024], tf.float32),
            "context": tf.TensorSpec([None, 12], tf.float32),
            "modality_mask": tf.TensorSpec([None, 3], tf.float32),
        }
        return (
            tf.data.Dataset.from_tensors(
                (
                    {
                        "image_embedding": np.zeros((1, 576), np.float32),
                        "audio_embedding": np.zeros((1, 1024), np.float32),
                        "context": np.zeros((1, 12), np.float32),
                        "modality_mask": np.ones((1, 3), np.float32),
                    },
                    np.zeros((1,), np.int32),
                    np.ones((1,), np.float32),
                )
            ).take(0)
        )

    ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "image_embedding": data["image_embedding"].astype(np.float32),
                "audio_embedding": data["audio_embedding"].astype(np.float32),
                "context": data["context"].astype(np.float32),
                "modality_mask": data["modality_mask"].astype(np.float32),
            },
            data["vibe_label"].astype(np.int32),
            data["sample_weight"].astype(np.float32),
        )
    )
    if training:
        ds = ds.shuffle(min(n, 4096), seed=seed, reshuffle_each_iteration=True)

        def _aug(batch_x, y, w):
            # batch not yet — apply per-example then batch
            return batch_x, y, w

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=training and n >= batch_size)
    if training and modality_dropout_rate > 0:

        def _drop(x, y, w):
            x2 = modality_dropout(x, modality_dropout_rate, training=True)
            return x2, y, w

        ds = ds.map(_drop, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights_from_labels(labels: np.ndarray, num_classes: int = 7) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / np.sqrt(counts)
    inv = inv * num_classes / inv.sum()
    return inv.astype(np.float32)


def metrics_report(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y_pred = probs.argmax(axis=-1)
    n = len(y_true)
    if n == 0:
        return {"macro_f1": 0.0, "accuracy": 0.0, "nll": 0.0}
    acc = float((y_pred == y_true).mean())
    # macro-F1
    f1s = []
    for c in range(probs.shape[-1]):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    macro_f1 = float(np.mean(f1s))
    # NLL
    p = probs[np.arange(n), y_true.astype(int)]
    nll = float(-np.log(np.clip(p, 1e-9, 1.0)).mean())
    return {"macro_f1": macro_f1, "accuracy": acc, "nll": nll}


def evaluate(model, data: Dict[str, np.ndarray]) -> Dict[str, float]:
    import tensorflow as tf

    n = data["vibe_label"].shape[0]
    if n == 0:
        return {"macro_f1": 0.0, "accuracy": 0.0, "nll": 0.0}
    x = {
        "image_embedding": tf.constant(data["image_embedding"]),
        "audio_embedding": tf.constant(data["audio_embedding"]),
        "context": tf.constant(data["context"]),
        "modality_mask": tf.constant(data["modality_mask"]),
    }
    out = model(x, training=False)
    probs = out["vibe_probabilities"].numpy()
    return metrics_report(data["vibe_label"], probs)


def train(args: argparse.Namespace) -> int:
    _setup_gpu(args.cpu)
    import tensorflow as tf
    from tensorflow import keras

    from ml.fusion_v0 import NUM_VIBES, build_fusion_v0, count_trainable_params

    man = json.loads(Path(args.manifest).read_text())
    base = Path(args.manifest).parent
    train_d = load_npz(base / man["train_npz"])
    val_d = load_npz(base / man["validation_npz"])

    if train_d["vibe_label"].shape[0] == 0:
        print("ERROR: empty train split", file=sys.stderr)
        return 1

    tf.keras.utils.set_random_seed(args.seed)
    model = build_fusion_v0(dropout=args.dropout)
    print(f"trainable params: {count_trainable_params(model):,}")

    cw = class_weights_from_labels(train_d["vibe_label"], NUM_VIBES)
    # fold class weights into sample weights
    sw = train_d["sample_weight"] * cw[train_d["vibe_label"].astype(int)]
    train_d = dict(train_d)
    train_d["sample_weight"] = sw.astype(np.float32)

    train_ds = make_dataset(
        train_d,
        args.batch_size,
        training=True,
        seed=args.seed,
        modality_dropout_rate=args.modality_dropout,
    )
    # simple custom loop for sample weights
    lr_sched = keras.optimizers.schedules.CosineDecay(
        args.learning_rate,
        decay_steps=max(args.epochs * max(len(train_d["vibe_label"]) // args.batch_size, 1), 1),
        alpha=0.05,
    )
    try:
        opt = keras.optimizers.AdamW(
            learning_rate=lr_sched, weight_decay=args.weight_decay
        )
    except AttributeError:
        opt = keras.optimizers.Adam(learning_rate=lr_sched)

    best_f1 = -1.0
    bad_epochs = 0
    history = []
    out_path = Path(args.weights_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    @tf.function
    def train_step(x, y, w):
        with tf.GradientTape() as tape:
            out = model(x, training=True)
            per = keras.losses.sparse_categorical_crossentropy(
                y, out["vibe_logits"], from_logits=True
            )
            aux = keras.losses.sparse_categorical_crossentropy(
                y, out["vibe_from_emb_logits"], from_logits=True
            )
            loss = tf.reduce_sum((per + 0.25 * aux) * w) / (tf.reduce_sum(w) + 1e-8)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(args.epochs):
        losses = []
        for x, y, w in train_ds:
            losses.append(float(train_step(x, y, w)))
        tr_loss = float(np.mean(losses)) if losses else 0.0
        val_m = evaluate(model, val_d)
        row = {"epoch": epoch + 1, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"epoch {epoch+1}/{args.epochs} loss={tr_loss:.4f} "
            f"val_macro_f1={val_m['macro_f1']:.3f} val_acc={val_m['accuracy']:.3f}"
        )
        if val_m["macro_f1"] >= best_f1:
            best_f1 = val_m["macro_f1"]
            bad_epochs = 0
            model.save_weights(str(out_path))
            print(f"  ↑ saved {out_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stop")
                break

    report = {
        "best_val_macro_f1": best_f1,
        "weights": str(out_path),
        "seed": args.seed,
        "manifest": str(args.manifest),
        "history": history,
        "class_weights": cw.tolist(),
    }
    rep_path = out_path.with_suffix(".train_report.json")
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"report → {rep_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Train fusion_v0")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--weights-out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--modality-dropout", type=float, default=0.15)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return train(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
