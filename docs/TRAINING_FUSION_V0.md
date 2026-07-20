# Training fusion_v0 (sole production entry)

## What you train

```text
Photo ── MobileNetV3Small ── [576] ┐
Audio ─────── YAMNet ─────── [1024]├─ fusion_v0 ─→ 7 vibe probs
context12-v1 ─────────────── [12]  │             └→ perceptual [128]
modality_mask ────────────── [3] ──┘
```

**Only documented production train command:** `python -m ml.train_fusion_v0`

| Path | Role |
|------|------|
| `ml/fusion_v0.py` | Dense production head |
| `ml/encoders.py` | Frozen MobileNetV3Small + YAMNet |
| `ml/context12.py` | context12-v1 (bit-compatible with Kotlin `ContextEncoderV1`) |
| `ml/tf_data_pipeline.py` | Experimental raw TFRecord parse/augment (MoE path) |
| `ml/prepare_fusion_dataset.py` | Gold → NPZ + manifest |
| `ml/validate_fusion_dataset.py` | Leakage / shape / schema |
| `ml/train_fusion_v0.py` | Official training entry |
| `ml/export_tflite.py` | TFLite for `FusionV0Interpreter` |
| `ml/evaluate_fusion_v0.py` | Held-out metrics |
| `ml/serve_fallback.py` | MoE → Ollama → Grok → rules (enrichment) |
| `backend/jobs/pyspark_export_gold.py` | Medallion Gold (pure Spark, no TF) |
| `ml/config.py` | `SCHEMA_VERSION` + constants |
| `ml/tests/test_fusion_v0.py` | Contract tests (must stay green) |
| `ml/notebooks/validate_fusion_live.py` | marimo exploration (not train) |

**Do not** use `ml/experiments/*` for release weights.

## tensorflow-talex training policy

```python
tf.keras.mixed_precision.set_global_policy("mixed_float16")
# final logits + loss stay float32

# production fusion_v0 currently trains from NPZ via train_fusion_v0;
# experimental TFRecord path (MoE) follows:
ds = (
    tf.data.Dataset.list_files(glob)
    .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)       # dequant + mask zeroing
    .map(cheap_augment, num_parallel_calls=tf.data.AUTOTUNE)  # SpecAugment-lite, Mixup, geo jitter
    .batch(global_batch, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)
```

Always:

- Gradient accumulation for effective large batch on one GPU (no MirroredStrategy)
- AdamW + cosine decay + warmup
- Class weights from manifest (sqrt-inverse, renormalized)
- Label smoothing 0.1, dropout ~0.15, early stop on val **macro-F1**
- Freeze MobileNet/YAMNet; train only fusion head
- No RLAIF until PolicyStageStatus + provenance allow; then ramp weight 0 → 0.3

### Optional MoE upgrade (same I/O)

Keep inputs/outputs identical. Replace middle FFN with top-k sparse MoE
(k=1–2, ≤8 experts) + load-balancing loss. Train from dense weights + LoRA.
Export still one TFLite the existing Kotlin interpreter can load. Code lives
under `ml/experiments/` until explicitly promoted.

## Local environment

```bash
# pyenv 3.12.x with TensorFlow 2.x + GPU if available
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python -m pip install -r ml/requirements-ml.txt
# optional exploration
python -m pip install marimo polars
```

## Pipeline

```bash
# 1) Export Train Mode JSONL (human labels + paths or precomputed embeddings)
#    consent_for_training=true only

# 2) Optional: PySpark medallion
#    python backend/jobs/pyspark_export_gold.py ...

# 3) Build NPZ (runs encoders unless --skip-encoders)
python -m ml.prepare_fusion_dataset \
  --input data/personal/train_mode_export.jsonl \
  --out-dir /secure/geojournal/fusion-v0

# 4) Validate splits
python -m ml.validate_fusion_dataset \
  --manifest /secure/geojournal/fusion-v0/manifest.json

# 5) Train
python -m ml.train_fusion_v0 \
  --manifest /secure/geojournal/fusion-v0/manifest.json \
  --weights-out /secure/artifacts/fusion-v0-seed42.weights.h5 \
  --epochs 60 --batch-size 32 --learning-rate 3e-4 \
  --modality-dropout 0.15 --seed 42

# 6) Evaluate held-out test NPZ
python -m ml.evaluate_fusion_v0 \
  --weights /secure/artifacts/fusion-v0-seed42.weights.h5 \
  --npz /secure/geojournal/fusion-v0/splits/test.npz

# 7) Export TFLite (float32 first; signature must match Kotlin)
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

## marimo (prep / validate only)

```bash
marimo edit ml/notebooks/validate_fusion_live.py
```

Change Bronze/Gold path or filters → only downstream cells re-run. Cache heavy
Polars results as parquet. **Do not** run the production train loop inside marimo.

## Split rules

- Group by `session_id` (≈60 min gap)
- Temporal: oldest 70% train / 15% val / 15% test
- Assert session hashes disjoint
- Augmentations stay in original split

## Baselines to beat

Majority class, logistic regression on concat features, image-only, audio-only,
context-only, image+context. Report **macro-F1**, per-class recall, NLL, ECE —
not accuracy alone.

## After v0

Valence/arousal multi-task, active learning, supervised contrastive on 128-D,
temperature scaling, location dropout, public sound as **masked aux loss only**,
sparse MoE only if dense capacity is proven insufficient.

See [CONFIRMATION.md](CONFIRMATION.md), [DATASETS.md](DATASETS.md),
[adr/ADR-003-fusion-v0-production.md](adr/ADR-003-fusion-v0-production.md).
