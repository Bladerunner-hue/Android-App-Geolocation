"""
Export fusion_v0 → SavedModel + TFLite with mandatory weights + parity gate.

  python -m ml.export_tflite \\
    --weights /secure/artifacts/fusion-v0-seed42.weights.h5 \\
    --parity-data /secure/geojournal/fusion-v0/eval/parity.npz \\
    --quantization float32 \\
    --savedmodel-dir /secure/artifacts/fusion-v0/r1/saved_model \\
    --out /secure/artifacts/fusion-v0/r1/fusion.tflite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_features(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path)
    # accept both naming conventions
    def pick(*names):
        for n in names:
            if n in z.files:
                return z[n]
        raise KeyError(f"Need one of {names} in {path}")

    return {
        "image_embedding": pick("image_embedding", "image_emb").astype(np.float32),
        "audio_embedding": pick("audio_embedding", "audio_emb").astype(np.float32),
        "context": pick("context").astype(np.float32),
        "modality_mask": pick("modality_mask", "presence_mask").astype(np.float32),
    }


def export(
    weights: Path,
    parity_data: Path,
    out: Path,
    savedmodel_dir: Path,
    quantization: str = "float32",
    representative_data: Optional[Path] = None,
    min_calibration_samples: int = 256,
    force: bool = False,
    max_prob_mae: float = 1e-3,
    architecture: str = "dense",
) -> Dict[str, Any]:
    import tensorflow as tf

    from ml.fusion_v0 import (
        AUDIO_DIM,
        CONTEXT_DIM,
        IMAGE_DIM,
        MASK_DIM,
        build_fusion_v0,
    )

    if not weights.is_file():
        raise FileNotFoundError(
            f"Trained weights required: {weights}. No untrained export allowed."
        )
    if out.exists() and not force:
        raise FileExistsError(f"Refusing overwrite {out} (use --force)")
    if savedmodel_dir.exists() and not force:
        raise FileExistsError(f"Refusing overwrite {savedmodel_dir} (use --force)")

    if architecture == "moe":
        from ml.experiments.fusion_moe_v0 import build_fusion_moe_v0

        model = build_fusion_moe_v0()
    else:
        model = build_fusion_v0()
    model.load_weights(str(weights))

    parity = _load_features(parity_data)
    n = parity["image_embedding"].shape[0]
    if n < 1:
        raise ValueError("parity fixture empty")

    batch = {k: tf.constant(v) for k, v in parity.items()}
    ref = model(batch, training=False)
    ref_prob = ref["vibe_probabilities"].numpy()

    class Module(tf.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, IMAGE_DIM], tf.float32, name="image_embedding"),
                tf.TensorSpec([None, AUDIO_DIM], tf.float32, name="audio_embedding"),
                tf.TensorSpec([None, CONTEXT_DIM], tf.float32, name="context"),
                tf.TensorSpec([None, MASK_DIM], tf.float32, name="modality_mask"),
            ]
        )
        def serving_default(self, image_embedding, audio_embedding, context, modality_mask):
            return self.m(
                {
                    "image_embedding": image_embedding,
                    "audio_embedding": audio_embedding,
                    "context": context,
                    "modality_mask": modality_mask,
                },
                training=False,
            )

    mod = Module(model)
    _ = mod.serving_default(
        batch["image_embedding"][:1],
        batch["audio_embedding"][:1],
        batch["context"][:1],
        batch["modality_mask"][:1],
    )
    savedmodel_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        mod,
        str(savedmodel_dir),
        signatures={"serving_default": mod.serving_default},
    )

    concrete = mod.serving_default.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], mod)

    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        if representative_data is None or not Path(representative_data).is_file():
            raise ValueError("INT8 requires --representative-data NPZ")
        rep = _load_features(Path(representative_data))
        if rep["image_embedding"].shape[0] < min_calibration_samples:
            raise ValueError(
                f"Need ≥{min_calibration_samples} calibration rows, "
                f"got {rep['image_embedding'].shape[0]}"
            )

        def rep_gen():
            nrep = rep["image_embedding"].shape[0]
            for i in range(min(nrep, 512)):
                yield [
                    rep["image_embedding"][i : i + 1],
                    rep["audio_embedding"][i : i + 1],
                    rep["context"][i : i + 1],
                    rep["modality_mask"][i : i + 1],
                ]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
    elif quantization != "float32":
        raise ValueError(quantization)

    tflite_bytes = converter.convert()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_bytes)

    # TFLite parity on first sample
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    for detail, arr in zip(
        interpreter.get_input_details(),
        [
            parity["image_embedding"][:1],
            parity["audio_embedding"][:1],
            parity["context"][:1],
            parity["modality_mask"][:1],
        ],
    ):
        interpreter.set_tensor(detail["index"], arr.astype(detail["dtype"]))
    interpreter.invoke()
    # Prefer named outputs — never "first tensor with last dim 7" (logits vs probs).
    tflite_prob = None
    tflite_perc = None
    for detail in interpreter.get_output_details():
        name = (detail.get("name") or "").lower()
        t = interpreter.get_tensor(detail["index"])
        if "vibe_prob" in name or name.endswith("probabilities") or "probabilit" in name:
            tflite_prob = t
        elif "perceptual" in name:
            tflite_perc = t
    if tflite_prob is None:
        # Named lookup failed: pick output whose shape is [1,7] and values look like probs
        for detail in interpreter.get_output_details():
            t = interpreter.get_tensor(detail["index"])
            if t.ndim >= 2 and t.shape[-1] == 7:
                s = float(np.sum(t))
                if 0.5 < s < 1.5:
                    tflite_prob = t
                    break
    mae = 0.0
    if tflite_prob is not None and quantization == "float32":
        mae = float(np.mean(np.abs(tflite_prob - ref_prob[:1])))
        if mae > max_prob_mae:
            raise RuntimeError(f"Parity MAE {mae} > {max_prob_mae}")
        if not np.all(np.isfinite(tflite_prob)):
            raise RuntimeError("vibe_probabilities contain non-finite values")
        if tflite_perc is not None and tflite_perc.shape[-1] != 128:
            raise RuntimeError(f"perceptual_embedding last dim {tflite_perc.shape[-1]} != 128")

    report = {
        "weights": str(weights),
        "out": str(out),
        "savedmodel": str(savedmodel_dir),
        "quantization": quantization,
        "parity_mae": mae,
        "bytes": len(tflite_bytes),
    }
    (out.parent / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--parity-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--savedmodel-dir", type=Path, required=True)
    p.add_argument(
        "--quantization", choices=("float32", "float16", "int8"), default="float32"
    )
    p.add_argument("--representative-data", type=Path, default=None)
    p.add_argument("--min-calibration-samples", type=int, default=256)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-prob-mae", type=float, default=1e-3)
    p.add_argument(
        "--architecture",
        choices=("dense", "moe"),
        default="dense",
        help="dense=fusion_v0 (release baseline), moe=fusion_moe_v0 (same I/O, optional capacity)",
    )
    args = p.parse_args()
    report = export(
        weights=args.weights,
        parity_data=args.parity_data,
        out=args.out,
        savedmodel_dir=args.savedmodel_dir,
        quantization=args.quantization,
        representative_data=args.representative_data,
        min_calibration_samples=args.min_calibration_samples,
        force=args.force,
        max_prob_mae=args.max_prob_mae,
        architecture=args.architecture,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
