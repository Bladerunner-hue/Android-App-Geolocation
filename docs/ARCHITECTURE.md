# GeoJournal architecture

## Boundaries

- TensorFlow does not run inside Spark ETL or the FastAPI request path.
- TFLite returns tensors; Kotlin maps them to typed evidence.
- Classifier exposes **vibe probabilities + perceptual embedding**, not chain-of-thought.
- Text semantic search uses a **768-D** multilingual encoder; fusion perceptual is **128-D**.
- Private Mode defaults on; cloud sync and enrichment are separate opt-ins.
- API failure never silently sends media/GPS to a cloud LLM.

## Capture → journal

```text
Photo + optional 10s mono 16 kHz WAV + optional location
  → Room memory (offline first)
  → optional Train Mode human label (before model reveal)
  → optional on-device fusion_v0 (if TFLite packaged)
  → WorkManager sync only if Private Mode OFF and cloud sync ON
  → FastAPI JWT-isolated storage + optional pgvector
```

## ML production vs experiment

| Path | Status |
|------|--------|
| `ml/fusion_v0.py` + frozen MobileNet/YAMNet | **Production** |
| `moe_kickstart.py` / `ml/train.py` synthetic | Experiment only |

## Context contract

`context12-v1` — see `ml/context12.py` and `ContextEncoderV1.kt`.

## Data plane (upcoming): PySpark

```text
Events / Train Mode export
  → PySpark Bronze → Silver → Gold
  → NPZ / TFRecord + manifest (executor-side writes)
  → pyenv TensorFlow train_fusion_v0
  → TFLite → Android
```

- **PySpark** is the planned bulk ETL entry (pure DataFrame/SQL, windowed sessions).
- No TensorFlow inside Spark ETL modules; no training the fusion head on executors.
- Optional Scala jobs can follow later if needed; they are not the default.
- Seed: `backend/jobs/pyspark_export_gold.py`.

Status and feedback: [CONFIRMATION.md](CONFIRMATION.md).
