# GeoJournal architecture

## Role split (no redundancy)

| Plane | Owns |
|-------|------|
| **Kotlin / Room** | Offline journal, privacy, Train Mode, edge fusion telemetry, sync outbox |
| **FastAPI + PG16/pgvector** | Auth, media, memories, training_labels, vector search storage |
| **pyenv TF + PySpark** | Offline train from consented labels → TFLite back to edge |

Kotlin is **edge capture + telemetry**, not a second trainer or 768-D encoder.

## Boundaries

- TensorFlow does not run inside Spark ETL or the FastAPI request path.
- TFLite returns tensors; Kotlin maps them to typed evidence.
- Classifier exposes **vibe probabilities + perceptual embedding**, not chain-of-thought.
- Text semantic search uses a **768-D** multilingual encoder; fusion perceptual is **128-D**.
- Private Mode defaults on; cloud sync and enrichment are separate opt-ins.
- API failure never silently sends media/GPS to a cloud LLM.
- **Single JWT store:** `TokenStore` → `AuthInterceptor` (no dual DataStore/Room token).
- Schema: `001_memories_pgvector.sql` + `002_ai_ml_alignment.sql`.

## Capture → journal

```text
Photo + optional 10s mono 16 kHz WAV + optional location
  → ContextEncoderV1 + optional fusion_v0 telemetry
  → Room memory (offline first; structured_evidence + perceptual JSON)
  → optional Train Mode human label (before model reveal) → Room training_labels
  → WorkManager outbox if !private && cloud sync && JWT
       POST /api/memories/analyze  (client tensors only; never invent vibe)
       POST /api/training/labels   (consent_for_cloud required)
  → FastAPI + pgvector store
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
