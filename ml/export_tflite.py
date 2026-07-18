"""
Post-training INT8 TFLite export for on-device vibe tagging (pass-3 style).

Representative dataset from TFRecord manifest (no random noise for calib).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.config import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_TFRECORD_DIR,
    DEFAULT_TFLITE_PATH,
)
from ml.tf_data_pipeline import build_dataset  # noqa: E402
from moe_kickstart import build_geoai_moe, setup_runtime  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "best.weights.h5",
    )
    p.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_TFRECORD_DIR / "train" / "manifest_train.json",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_TFLITE_PATH)
    p.add_argument("--num-calib", type=int, default=32)
    args = p.parse_args()

    setup_runtime(mixed_precision=False)
    model = build_geoai_moe(hidden=256, num_blocks=2)

    if args.weights.exists():
        model.load_weights(str(args.weights))
        print(f"Loaded {args.weights}")
    else:
        print("WARNING: no weights — exporting untrained structure for pipeline test")

    # Concrete TF function for converter
    @tf.function(
        input_signature=[
            tf.TensorSpec([1, 224, 224, 3], tf.float32),
            tf.TensorSpec([1, 96, 64, 1], tf.float32),
            tf.TensorSpec([1, 8], tf.float32),
        ]
    )
    def serving(image, audio_mel, geo):
        out = model.infer(image, audio_mel, geo)
        # TFLite prefers flat tensors
        return {
            "vibe_id": out["vibe_id"],
            "vibe_prob": out["vibe_prob"],
            "insight_embedding": out["insight_embedding"],
        }

    concrete = serving.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)

    if args.train_manifest.exists():
        calib_ds = build_dataset(
            args.train_manifest, batch_size=1, training=False, cache=False
        )

        def rep():
            n = 0
            for x, _ in calib_ds:
                yield [
                    tf.cast(x["image"], tf.float32),
                    tf.cast(x["audio_mel"], tf.float32),
                    tf.cast(x["geo"], tf.float32),
                ]
                n += 1
                if n >= args.num_calib:
                    break

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep
        try:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        except Exception:
            pass
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(tflite_model)
    print(f"TFLite → {args.out} ({len(tflite_model)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
