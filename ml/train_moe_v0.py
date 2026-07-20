"""
Train fusion_moe_v0 on the same NPZ/manifest as fusion_v0.

  python -m ml.train_moe_v0 \\
    --manifest path/to/manifest.json \\
    --weights-out ml/artifacts/fusion-moe-v0.weights.h5 \\
    --epochs 30 --batch-size 32 --seed 42

Dense fusion_v0 remains the release baseline until MoE wins held-out macro-F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def main() -> None:
    p = argparse.ArgumentParser(description="Train fusion_moe_v0")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--weights-out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--init-from-dense", type=Path, default=None)
    args = p.parse_args()

    _setup_gpu(args.cpu)
    import tensorflow as tf
    from tensorflow import keras

    from ml.fusion_moe_v0 import build_fusion_moe_v0
    from ml.fusion_v0 import count_trainable_params
    from ml.train_fusion_v0 import load_npz, make_dataset

    tf.keras.utils.set_random_seed(args.seed)
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

    manifest = json.loads(args.manifest.read_text())
    train_path = Path(
        manifest.get("train_npz")
        or manifest.get("train")
        or manifest.get("splits", {}).get("train", {}).get("npz", "")
    )
    val_path = Path(
        manifest.get("val_npz")
        or manifest.get("val")
        or manifest.get("splits", {}).get("val", {}).get("npz", train_path)
    )
    if not train_path.is_file():
        # wiring fixture layout
        root = args.manifest.parent
        candidates = list(root.glob("**/*train*.npz")) + list(root.glob("**/*.npz"))
        if not candidates:
            raise SystemExit(f"No train NPZ found from manifest {args.manifest}")
        train_path = candidates[0]
        val_path = candidates[0]

    train_data = load_npz(train_path)
    val_data = load_npz(val_path)
    train_ds = make_dataset(
        train_data, args.batch_size, training=True, seed=args.seed, modality_dropout_rate=0.1
    )
    val_ds = make_dataset(
        val_data, args.batch_size, training=False, seed=args.seed, modality_dropout_rate=0.0
    )

    model = build_fusion_moe_v0(num_experts=args.num_experts, top_k=args.top_k)
    if args.init_from_dense and args.init_from_dense.is_file():
        from ml.fusion_v0 import build_fusion_v0

        dense = build_fusion_v0()
        dense.load_weights(str(args.init_from_dense), by_name=True, skip_mismatch=True)
        dmap = {w.name: w for w in dense.weights}
        n_copy = 0
        for w in model.weights:
            if w.name in dmap and tuple(w.shape) == tuple(dmap[w.name].shape):
                w.assign(dmap[w.name])
                n_copy += 1
        print(f"Warm-started {n_copy} tensors from dense fusion_v0")

    print("trainable params", count_trainable_params(model))
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = keras.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4)

    @tf.function
    def train_step(batch, y, sw):
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            logits = tf.cast(out["vibe_logits"], tf.float32)
            main = loss_fn(y, logits, sample_weight=sw)
            emb_logits = tf.cast(out["vibe_from_emb_logits"], tf.float32)
            aux = loss_fn(y, emb_logits, sample_weight=sw)
            # SparseMoEBlock add_loss is in model.losses
            bal = tf.add_n(model.losses) if model.losses else tf.constant(0.0, tf.float32)
            loss = main + 0.3 * aux + bal
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best = -1.0
    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        losses = []
        for batch, y, sw in train_ds:
            losses.append(float(train_step(batch, y, sw)))
        correct = total = 0
        for batch, y, sw in val_ds:
            out = model(batch, training=False)
            pred = tf.argmax(tf.cast(out["vibe_logits"], tf.float32), axis=-1, output_type=tf.int32)
            correct += int(tf.reduce_sum(tf.cast(pred == y, tf.int32)))
            total += int(tf.shape(y)[0])
        acc = correct / max(total, 1)
        print(f"epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.4f} val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            model.save_weights(str(args.weights_out))
            print(f"  saved {args.weights_out}")

    meta = {
        "model": "fusion_moe_v0",
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "best_val_acc": best,
        "weights": str(args.weights_out),
        "manifest": str(args.manifest),
        "note": "Release only if val macro-F1 beats dense fusion_v0 on same held-out sessions",
    }
    args.weights_out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print("done", meta)


if __name__ == "__main__":
    main()
