# GeoJournal — confirmation + feedback log

Living status of the production ML path. Update when environment, data, or
pipeline gates change.

## Feedback integrated (2026-07-20)

Vertical-slice priority over Gemini Nano / Kalman / AR:

| Item | Status |
|------|--------|
| WorkManager + Hilt-Work + TFLite Gradle deps | Done |
| CAMERA / RECORD_AUDIO + FileProvider | Done |
| Capture location permission state + accuracy passthrough | Done |
| modality_mask = [photo, audio, time] via `ContextEncoderV1.modalityMask` | Done |
| Single `EdgeMemoryAnalyzer` (fusion_v0 asset name) | Done |
| Search mode never claims `semantic` (lexical / lexical_placeholder) | Done |
| Stable text-hash (blake2b), not Python `hash()` | Done |
| Media stream + size bound + magic sniff | Done |
| `DELETE /api/memories/{id}` | Done |
| CI: backend pytest + assembleDebug | Done |
| Real pgvector SQL + PostGIS | Deferred (schema ready; app still JSON/SQLite-friendly) |
| CameraX rewrite | Deferred until capture path is green end-to-end |

## Sprint (bronze export + encoders scaffold) 2026-07-20

| Item | Status |
|------|--------|
| `TrainingBronzeExporter` (consent_for_training → bronze JSONL + zip) | Done — Privacy screen button |
| `ImageEncoderTFLite` / `AudioEncoderTFLite` load assets if present | Done — honest unavailable without assets |
| `extractorsReady` flips when both encoder assets package | Done |
| `ml/export_encoders_tflite.py` | Done — MobileNet (+ YAMNet if hub available) |
| `insightEmbeddingJson` Room v5 | Done |
| Journal uses `EdgeMemoryAnalyzer` | Done |

## Confirmed (this machine)

| Item | Status |
|------|--------|
| Production model | **Dense `fusion_v0`** — not experimental MoE |
| Runtime | **pyenv Python 3.12** + **TensorFlow 2.20.0** (ready; no extra venv required) |
| GPU | **NVIDIA RTX A5000** visible to TF |
| Head size | ~**284k** trainable params |
| Wiring train | `ml/artifacts/wiring-seed42.weights.h5` — val macro-F1 ≈ 0.59 on synthetic features (pipeline only, not accuracy claim) |
| Contracts | `context12-v1`, modality mask [3], NPZ schema, session-grouped splits |
| Android | Train Mode entity, `ContextEncoderV1`, `FusionV0Interpreter` (unavailable without packaged TFLite) |
| Weights in git | **No** — artifacts ignored |

### Commands that worked

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python -m ml.make_wiring_fixture
python -m ml.validate_fusion_dataset --manifest ml/data_sample/fusion_wiring/manifest.json
python -m ml.train_fusion_v0 \
  --manifest ml/data_sample/fusion_wiring/manifest.json \
  --weights-out ml/artifacts/wiring-seed42.weights.h5 \
  --epochs 8 --batch-size 16 --learning-rate 1e-3 --seed 42
```

## Feedback accepted (product / ML rules)

1. **Train vibes only from human Train Mode labels** — never invent 7-class vibes from ESC/FSD/images.
2. **Three model spaces stay separate:**
   - `fusion_v0` → vibe probs [7] + perceptual [128]
   - E5 text encoder → semantic [1024] + pgvector (`intfloat/e5-large-v2` @ :6100)
   - optional sound-event head → factual events (rain/speech/traffic)
3. **Frozen encoders first:** MobileNetV3Small [576], YAMNet mean-pool [1024].
4. **Public data policy:** ESC-10 benchmark (CC BY); ESC-50 full noncommercial; FSD50K start CC0; image-emotion sets skip for v0.
5. **Train Mode anti-anchoring:** user labels **before** model prediction is shown.
6. **Leakage-safe splits:** session rotation (~60 min), temporal 70/15/15, disjoint session hashes.
7. **Export gates:** trained weights required; float32 TFLite first; INT8 only with real representative NPZ + parity; held-out metrics beyond numerical parity.
8. **Local A5000 default** for feature extract + head train; HF Jobs only after branch/image publish; no paid Jobs launched yet from this log.
9. **HF private only:** deidentified embeddings/labels/provenance — never raw media, GPS precision, Room DBs, tokens.
10. **`ml/train.py` / synthetic MoE** = experiment only, not release.

## Still open (gates)

| Gate | Note |
|------|------|
| Human Train Mode volume | Target wiring ~30/class; internal v0 ~100–150/class |
| Publish production branch | Public GitHub `master` may lag; cloud Jobs need published code or image digest |
| On-device encoder TFLite | MobileNet + YAMNet on Android for full edge path |
| Semantic 1024 E5 service | Direct HTTP :6100 `intfloat/e5-large-v2` (OpenClaw bridge :6200) |
| Multi-seed sweeps | 42 / 1337 / 2026 + LR / dropout grid — not on test set |
| Baselines | Majority, logistic, image-only, audio-only, context-only |

## Upcoming: PySpark (not Scala-first)

**Data plane will leverage PySpark next** for medallion Bronze → Silver → Gold,
session features, license-safe sound bootstrap manifests, and executor-side
export to training NPZ / TFRecords.

| Rule | Detail |
|------|--------|
| Role | Ingestion, joins, windowed trajectory/session features, class counts, manifests |
| **Not** | Training the fusion head inside Spark; no TensorFlow in pure ETL modules |
| Export | Gold → NPZ/TFRecord + `manifest.json` (row counts, hashes, class weights) |
| Driver safety | Never `collect()` raw photo/audio bytes to the driver |
| JVM path | Prefer pure DataFrame/SQL + window functions; optional Scala jobs later if needed — **PySpark is the planned entry** |
| Existing seed | `backend/jobs/pyspark_export_gold.py` (local stub + `--spark` path) |

```text
App JSONL / events
  → PySpark Bronze / Silver / Gold
  → prepare_fusion_dataset style features (or Spark-written NPZ shards)
  → pyenv TF train_fusion_v0 (A5000)
  → export_tflite → Android asset
```

## After v0 (priority)

1. Valence + arousal multi-task  
2. Active learning (entropy / rare class)  
3. Supervised contrastive on 128-D  
4. Per-user calibration  
5. Temperature scaling + unclassified threshold  
6. Train-only raw augment before embed  
7. Missing-modality eval matrix  
8. Location dropout / coarse geo  
9. Public sound as **masked aux loss only**  
10. Structured evidence (not CoT theater)  
11. RLAIF only for optional captions — not vibe truth  
12. Sparse MoE only if dense capacity is exhausted  

## Related docs

- [TRAINING_FUSION_V0.md](TRAINING_FUSION_V0.md)
- [DATASETS.md](DATASETS.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [adr/ADR-003-fusion-v0-production.md](adr/ADR-003-fusion-v0-production.md)
