"""Export frozen MobileNetV3Small (+ optional YAMNet wrapper) to TFLite for Android.

Usage (pyenv 3.12 + TF):
  python -m ml.export_encoders_tflite --out-dir app/src/main/assets

Does NOT invent fusion weights. Place resulting files next to fusion_v0.tflite.
YAMNet export requires tensorflow-hub; if missing, only MobileNet is written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def export_mobilenet(out: Path) -> None:
    from ml.encoders import IMAGE_DIM, ImageEncoder

    enc = ImageEncoder()
    # Concrete function: jpeg bytes → embedding (Android may also feed float NHWC)
    @tf.function(
        input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="image")]
    )
    def serving(image: tf.Tensor) -> tf.Tensor:
        # image already [0,255] float — model has include_preprocessing=True
        return enc.model(image, training=False)

    concrete = serving.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], serving)
    converter.optimizations = []
    data = converter.convert()
    out.write_bytes(data)
    # smoke
    interp = tf.lite.Interpreter(model_content=data)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    dummy = np.zeros(inp["shape"], dtype=np.float32)
    interp.set_tensor(inp["index"], dummy)
    interp.invoke()
    out_d = interp.get_output_details()[0]
    vec = interp.get_tensor(out_d["index"])
    assert vec.shape[-1] == IMAGE_DIM, vec.shape
    print(f"Wrote {out} shape={vec.shape}")


def export_yamnet_meanpool(out: Path) -> bool:
    try:
        from ml.encoders import AUDIO_DIM, AudioEncoder
    except Exception as exc:  # noqa: BLE001
        print(f"Skip YAMNet: {exc}")
        return False
    try:
        enc = AudioEncoder()
    except Exception as exc:  # noqa: BLE001
        print(f"Skip YAMNet load: {exc}")
        return False

    # Fixed window: 1 second @ 16 kHz for a stable TFLite graph (callers can window/mean later)
    win = 16_000

    @tf.function(input_signature=[tf.TensorSpec([win], tf.float32, name="waveform")])
    def serving(waveform: tf.Tensor) -> tf.Tensor:
        return enc.embed_waveform(waveform)

    concrete = serving.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], serving)
    try:
        data = converter.convert()
    except Exception as exc:  # noqa: BLE001
        print(f"YAMNet TFLite convert failed: {exc}")
        return False
    out.write_bytes(data)
    print(f"Wrote {out} (window={win}, dim={AUDIO_DIM})")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("app/src/main/assets"),
        help="Android assets directory",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_mobilenet(args.out_dir / "mobilenet_v3_small.tflite")
    export_yamnet_meanpool(args.out_dir / "yamnet_meanpool.tflite")
    print("Done. Assets are gitignored if large; package for release builds.")


if __name__ == "__main__":
    main()
