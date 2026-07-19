# Training fusion_v0 (production path)

## What you train

```text
Photo ── MobileNetV3Small ── [576] ┐
Audio ─────── YAMNet ─────── [1024]├─ fusion_v0 ─→ 7 vibe probs
context12-v1 ─────────────── [12]  │             └→ perceptual [128]
modality_mask ────────────── [3] ──┘
```

Files:

| File | Role |
|------|------|
| `ml/fusion_v0.py` | Dense head |
| `ml/context12.py` | Context contract |
| `ml/encoders.py` | Frozen MobileNet + YAMNet |
| `ml/prepare_fusion_dataset.py` | JSONL → NPZ splits |
| `ml/validate_fusion_dataset.py` | Leakage / shape checks |
| `ml/train_fusion_v0.py` | Train head |
| `ml/export_tflite.py` | SavedModel + TFLite |
| `ml/evaluate_fusion_v0.py` | Held-out metrics |
| `ml/sound_bootstrap.py` | ESC/FSD event lists (not vibes) |

**Do not** use `ml/train.py` / synthetic MoE for release.

## Local environment

Use the existing pyenv TensorFlow environment if it already has TF 2.x + GPU
(verified: TF 2.20 + A5000 works). Only create a venv if you need isolation
from PyTorch or a clean pin:

```bash
# already-ready pyenv
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"

# optional: tensorflow-hub / soundfile if missing
python -m pip install -r ml/requirements-ml.txt
```

## Pipeline

```bash
# 1) Export Train Mode JSONL (human labels + paths or precomputed embeddings)

# 2) Build NPZ (runs encoders unless --skip-encoders)
python -m ml.prepare_fusion_dataset \
  --input data/personal/train_mode_export.jsonl \
  --out-dir /secure/geojournal/fusion-v0

# 3) Validate splits
python -m ml.validate_fusion_dataset \
  --manifest /secure/geojournal/fusion-v0/manifest.json

# 4) Train (A5000 / local GPU; --cpu if needed)
python -m ml.train_fusion_v0 \
  --manifest /secure/geojournal/fusion-v0/manifest.json \
  --weights-out /secure/artifacts/fusion-v0-seed42.weights.h5 \
  --epochs 60 --batch-size 32 --learning-rate 3e-4 \
  --modality-dropout 0.15 --seed 42

# 5) Evaluate held-out test NPZ
python -m ml.evaluate_fusion_v0 \
  --weights /secure/artifacts/fusion-v0-seed42.weights.h5 \
  --npz /secure/geojournal/fusion-v0/splits/test.npz

# 6) Export TFLite (float32 first)
python -m ml.export_tflite \
  --weights /secure/artifacts/fusion-v0-seed42.weights.h5 \
  --parity-data /secure/geojournal/fusion-v0/eval/parity.npz \
  --quantization float32 \
  --savedmodel-dir /secure/artifacts/fusion-v0/r1/saved_model \
  --out /secure/artifacts/fusion-v0/r1/fusion.tflite \
  --force
```

## Wiring smoke (no personal data)

```bash
python -m ml.make_wiring_fixture
python -m ml.validate_fusion_dataset --manifest ml/data_sample/fusion_wiring/manifest.json
python -m ml.train_fusion_v0 \
  --manifest ml/data_sample/fusion_wiring/manifest.json \
  --weights-out ml/artifacts/wiring.weights.h5 \
  --epochs 5 --batch-size 16 --cpu
```

## Baselines to beat

Majority class, logistic regression on concat features, image-only, audio-only,
context-only, image+context. Report **macro-F1**, per-class recall, NLL, ECE —
not accuracy alone.

## Split rules

- Group by `session_id` (≈60 min gap)
- Temporal: oldest 70% train / 15% val / 15% test
- Assert session hashes disjoint
- Augmentations stay in original split

## After v0

Valence/arousal multi-task, active learning, supervised contrastive on 128-D,
temperature scaling, location dropout, public sound as **masked aux loss only**,
sparse MoE only if dense capacity is proven insufficient.

## Upcoming data plane: PySpark

Feature medallion and bulk export will use **PySpark** (not a Scala-first
rewrite). TF training stays in pyenv; Spark never runs the fusion head.

See [CONFIRMATION.md](CONFIRMATION.md) for the accepted feedback log and
PySpark boundary rules. Seed job: `backend/jobs/pyspark_export_gold.py`.
