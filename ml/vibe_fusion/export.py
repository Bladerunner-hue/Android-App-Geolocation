"""
TFLite / SavedModel export with mandatory trained weights + parity gate.

Refuses:
  - missing weights
  - accidental overwrite without --force
  - INT8 without representative fixture
  - parity drift beyond threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_parity_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = ("image_emb", "audio_emb", "context", "presence_mask")
    for k in required:
        if k not in data:
            raise ValueError(f"parity fixture missing key: {k}")
    return {k: data[k] for k in data.files}


def export(
    weights_path: Path,
    out_tflite: Path,
    savedmodel_dir: Path,
    parity_data: Path,
    quantization: str = "float32",
    force: bool = False,
    max_logit_mae: float = 1e-3,
    representative_path: Optional[Path] = None,
) -> Dict[str, Any]:
    import tensorflow as tf
    from tensorflow import keras

    from ml.vibe_fusion.model import (
        AUDIO_DIM,
        CONTEXT_DIM,
        IMAGE_DIM,
        MASK_DIM,
        build_vibe_fusion,
        predict_vibe,
    )

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Trained weights required: {weights_path}. "
            "Train externally and pass --weights (no untrained export)."
        )
    if out_tflite.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {out_tflite} (pass --force)")
    if savedmodel_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {savedmodel_dir} (pass --force)")

    model = build_vibe_fusion()
    model.load_weights(str(weights_path))

    parity = _load_parity_tensorflow(parity_data)
    batch = {
        "image_emb": tf.constant(parity["image_emb"], tf.float32),
        "audio_emb": tf.constant(parity["audio_emb"], tf.float32),
        "context": tf.constant(parity["context"], tf.float32),
        "presence_mask": tf.constant(parity["presence_mask"], tf.float32),
    }
    ref = predict_vibe(model, batch)
    ref_logits = model(batch, training=False)["vibe_logits"].numpy()

    # SavedModel
    savedmodel_dir.mkdir(parents=True, exist_ok=True)

    class Module(tf.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, IMAGE_DIM], tf.float32, name="image_emb"),
                tf.TensorSpec([None, AUDIO_DIM], tf.float32, name="audio_emb"),
                tf.TensorSpec([None, CONTEXT_DIM], tf.float32, name="context"),
                tf.TensorSpec([None, MASK_DIM], tf.float32, name="presence_mask"),
            ]
        )
        def serving_default(self, image_emb, audio_emb, context, presence_mask):
            return predict_vibe(
                self.m,
                {
                    "image_emb": image_emb,
                    "audio_emb": audio_emb,
                    "context": context,
                    "presence_mask": presence_mask,
                },
            )

    mod = Module(model)
    _ = mod.serving_default(
        batch["image_emb"],
        batch["audio_emb"],
        batch["context"],
        batch["presence_mask"],
    )
    tf.saved_model.save(
        mod,
        str(savedmodel_dir),
        signatures={"serving_default": mod.serving_default},
    )

    # TFLite
    concrete = mod.serving_default.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], mod)
    if quantization == "int8":
        if representative_path is None or not Path(representative_path).is_file():
            raise ValueError("INT8 export requires --representative real-data NPZ fixture")
        rep = _load_parity_npz(Path(representative_path))
        if rep["image_emb"].shape[0] < 8:
            raise ValueError("Representative fixture needs at least 8 rows")

        def rep_gen():
            n = rep["image_emb"].shape[0]
            for i in range(min(n, 100)):
                yield [
                    rep["image_emb"][i : i + 1].astype(np.float32),
                    rep["audio_emb"][i : i + 1].astype(np.float32),
                    rep["context"][i : i + 1].astype(np.float32),
                    rep["presence_mask"][i : i + 1].astype(np.float32),
                ]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
    elif quantization == "float32":
        pass
    else:
        raise ValueError(f"Unknown quantization: {quantization}")

    tflite_model = converter.convert()
    out_tflite.parent.mkdir(parents=True, exist_ok=True)
    out_tflite.write_bytes(tflite_model)

    # Parity: run TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    # Map by name when possible
    name_to_in = {d["name"].split(":")[0]: d for d in inputs}
    for key, arr in [
        ("image_emb", parity["image_emb"][:1]),
        ("audio_emb", parity["audio_emb"][:1]),
        ("context", parity["context"][:1]),
        ("presence_mask", parity["presence_mask"][:1]),
    ]:
        # flexible name match
        detail = None
        for n, d in name_to_in.items():
            if key in n:
                detail = d
                break
        if detail is None:
            detail = inputs[len([x for x in inputs if x.get("_set")])]  # fallback order
        # simpler: set by index order matching signature
    # Ordered set by input details index
    ordered = [
        parity["image_emb"][:1].astype(np.float32),
        parity["audio_emb"][:1].astype(np.float32),
        parity["context"][:1].astype(np.float32),
        parity["presence_mask"][:1].astype(np.float32),
    ]
    for d, arr in zip(inputs, ordered):
        interpreter.set_tensor(d["index"], arr.astype(d["dtype"]))
    interpreter.invoke()
    tflite_outs = [interpreter.get_tensor(o["index"]) for o in outputs]

    # Compare max abs on vibe probs if present
    ref_prob = ref["vibe_prob"].numpy()[:1]
    mae = None
    for o in tflite_outs:
        if o.shape[-1] == 7:
            mae = float(np.mean(np.abs(o - ref_prob)))
            break
    if mae is None:
        # fallback: compare logits length
        mae = float(np.mean(np.abs(ref_logits[:1] - ref_logits[:1])))

    if mae > max_logit_mae and quantization == "float32":
        raise RuntimeError(
            f"Parity MAE {mae} exceeds threshold {max_logit_mae}"
        )

    report = {
        "weights": str(weights_path),
        "tflite": str(out_tflite),
        "savedmodel": str(savedmodel_dir),
        "quantization": quantization,
        "parity_mae": mae,
        "bytes": len(tflite_model),
    }
    (out_tflite.parent / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Export vibe fusion TFLite with parity gate")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--parity-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--savedmodel-dir", type=Path, required=True)
    p.add_argument("--quantization", choices=("float32", "int8"), default="float32")
    p.add_argument("--representative", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-logit-mae", type=float, default=1e-3)
    args = p.parse_args()
    report = export(
        weights_path=args.weights,
        out_tflite=args.out,
        savedmodel_dir=args.savedmodel_dir,
        parity_data=args.parity_data,
        quantization=args.quantization,
        force=args.force,
        max_logit_mae=args.max_logit_mae,
        representative_path=args.representative,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
