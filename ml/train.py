"""
Train GeoAI MoE on minimal data (tensorflow-talex + manifest class weights).

Usage (from repo root):
  python -m ml.synthetic_bootstrap          # ~200 synthetic examples
  python -m ml.train --epochs 5             # quick functional model
  python -m ml.train --epochs 12 --export   # train + SavedModel

Real data: point --train-manifest / --val-manifest at PySpark Gold manifests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.config import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_SAVEDMODEL_DIR,
    DEFAULT_TFRECORD_DIR,
    TrainConfig,
)
from ml.tf_data_pipeline import build_dataset, class_weights_from_manifest  # noqa: E402
from moe_kickstart import (  # noqa: E402
    GradAccumTrainer,
    build_geoai_moe,
    make_optimizer,
    setup_runtime,
    total_train_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GeoAI multimodal MoE")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_TFRECORD_DIR / "train" / "manifest_train.json",
    )
    p.add_argument(
        "--val-manifest",
        type=Path,
        default=DEFAULT_TFRECORD_DIR / "val" / "manifest_val.json",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--export", action="store_true", help="Write SavedModel after train")
    p.add_argument("--savedmodel-dir", type=Path, default=DEFAULT_SAVEDMODEL_DIR)
    p.add_argument("--bootstrap", action="store_true", help="Generate synth data if missing")
    p.add_argument("--max-train-steps", type=int, default=0, help="0 = full epoch")
    p.add_argument("--cpu", action="store_true", help="Force CPU (skip GPU/cuDNN)")
    return p.parse_args()


def ensure_data(args: argparse.Namespace) -> None:
    if args.train_manifest.exists():
        return
    if not args.bootstrap and not args.train_manifest.exists():
        print("No TFRecords found — running synthetic bootstrap...")
    from ml.synthetic_bootstrap import bootstrap

    bootstrap()


@tf.function(reduce_retracing=True)
def _eval_batch(model, image, audio, geo, y_vibe, y_cot, class_weights):
    out = model(
        {"image": image, "audio_mel": audio, "geo": geo},
        training=False,
    )
    loss, parts = total_train_loss(
        y_vibe, y_cot, out, class_weights=class_weights
    )
    pred = tf.argmax(out["vibe_logits"], axis=-1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(y_vibe, pred.dtype)), tf.float32))
    parts = dict(parts)
    parts["acc"] = acc
    return parts


def evaluate(model, ds, class_weights) -> dict:
    totals = {}
    n = 0
    for x, y in ds:
        parts = _eval_batch(
            model, x["image"], x["audio_mel"], x["geo"], y["vibe"], y["cot"], class_weights
        )
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in totals.items()}


def export_savedmodel(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    class ServingModule(tf.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, 224, 224, 3], tf.float32, name="image"),
                tf.TensorSpec([None, 96, 64, 1], tf.float32, name="audio_mel"),
                tf.TensorSpec([None, 8], tf.float32, name="geo"),
            ]
        )
        def serving_default(self, image, audio_mel, geo):
            return self.model.infer(image, audio_mel, geo)

    mod = ServingModule(model)
    # Trace once
    _ = mod.serving_default(
        tf.zeros([1, 224, 224, 3]),
        tf.zeros([1, 96, 64, 1]),
        tf.zeros([1, 8]),
    )
    tf.saved_model.save(
        mod,
        str(path),
        signatures={"serving_default": mod.serving_default},
    )
    print(f"SavedModel → {path}")


def main() -> int:
    args = parse_args()
    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        accum_steps=args.accum_steps,
        hidden=args.hidden,
        mixed_precision=not args.no_mixed_precision,
    )
    if args.cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    setup_runtime(
        mixed_precision=cfg.mixed_precision and not args.cpu,
        force_cpu=args.cpu,
    )
    ensure_data(args)

    train_ds = build_dataset(
        args.train_manifest,
        batch_size=cfg.batch_size,
        training=True,
        shuffle_buffer=cfg.shuffle_buffer,
    )
    val_ds = build_dataset(
        args.val_manifest,
        batch_size=cfg.batch_size,
        training=False,
        cache=True,
    )
    class_weights = class_weights_from_manifest(args.train_manifest)
    print("class_weights:", class_weights.numpy())

    man = json.loads(args.train_manifest.read_text())
    n_train = int(sum(man.get("row_counts", [cfg.batch_size * 10])))
    steps_per_epoch = max(n_train // cfg.batch_size, 1)
    total_steps = steps_per_epoch * cfg.epochs

    model = build_geoai_moe(
        hidden=cfg.hidden,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        num_blocks=cfg.num_blocks,
        freeze_vision=cfg.freeze_vision,
        lora_rank=cfg.lora_rank,
    )
    opt = make_optimizer(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_steps,
    )
    trainer = GradAccumTrainer(
        model,
        opt,
        class_weights=class_weights,
        accum_steps=cfg.accum_steps,
        cot_weight=cfg.cot_weight,
        label_smoothing=cfg.label_smoothing,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    history = []

    for epoch in range(cfg.epochs):
        t0 = time.time()
        step = 0
        running = {}
        for x, y in train_ds:
            parts = trainer.train_step(
                x["image"], x["audio_mel"], x["geo"], y["vibe"], y["cot"]
            )
            for k, v in parts.items():
                if k == "applied":
                    continue
                running[k] = running.get(k, 0.0) + float(v)
            step += 1
            if args.max_train_steps and step >= args.max_train_steps:
                break
        train_metrics = {k: v / max(step, 1) for k, v in running.items()}
        val_metrics = evaluate(model, val_ds, class_weights)
        dt = time.time() - t0
        row = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "sec": dt,
        }
        history.append(row)
        print(
            f"epoch {epoch+1}/{cfg.epochs}  "
            f"train_loss={train_metrics.get('total_loss', 0):.4f}  "
            f"val_acc={val_metrics.get('acc', 0):.3f}  "
            f"val_loss={val_metrics.get('total_loss', 0):.4f}  "
            f"({dt:.1f}s)"
        )
        acc = val_metrics.get("acc", 0.0)
        if acc >= best_acc:
            best_acc = acc
            ckpt = args.checkpoint_dir / "best.weights.h5"
            model.save_weights(str(ckpt))
            print(f"  ↑ best weights → {ckpt}")

    (args.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))

    if args.export:
        # Reload best
        best = args.checkpoint_dir / "best.weights.h5"
        if best.exists():
            model.load_weights(str(best))
        export_savedmodel(model, args.savedmodel_dir)

    print("DONE. best val_acc=", best_acc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
