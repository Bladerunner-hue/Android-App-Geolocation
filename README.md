# GeoJournal

Offline-first personal geo memory journal: photo + optional ambient audio +
optional location → private Room journal → optional dense on-device vibe model →
optional authenticated backend with pgvector.

## Production ML path (this is the one)

```text
MobileNetV3Small [576] + YAMNet [1024] + context12 [12] + mask [3]
  → fusion_v0 dense head → 7 vibe probs + 128-D perceptual embedding
```

| Do | Do not |
|----|--------|
| `python -m ml.train_fusion_v0` | `python -m ml.train` (experimental MoE) |
| Human Train Mode labels | Fabricate vibes from ESC/FSD |
| Frozen YAMNet + MobileNet | Untrained weights in git |
| Session-grouped temporal splits | Random i.i.d. photo splits |

Docs: [docs/CONFIRMATION.md](docs/CONFIRMATION.md) (status + feedback) ·
[docs/TRAINING_FUSION_V0.md](docs/TRAINING_FUSION_V0.md) ·
[docs/DATASETS.md](docs/DATASETS.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

**Next data plane:** PySpark medallion → NPZ/manifest (TF stays in pyenv).

### Quick wiring (synthetic features only)

Uses your current pyenv TensorFlow (GPU if available):

```bash
python -m ml.make_wiring_fixture
python -m ml.validate_fusion_dataset --manifest ml/data_sample/fusion_wiring/manifest.json
python -m ml.train_fusion_v0 \
  --manifest ml/data_sample/fusion_wiring/manifest.json \
  --weights-out ml/artifacts/wiring-seed42.weights.h5 \
  --epochs 8 --batch-size 16 --learning-rate 1e-3 --seed 42
```

### Real training (local A5000)

1. Collect Train Mode labels (label **before** model reveal).
2. `python -m ml.prepare_fusion_dataset --input … --out-dir /secure/…`
3. `python -m ml.validate_fusion_dataset --manifest …/manifest.json`
4. `python -m ml.train_fusion_v0 --manifest … --weights-out … --seed 42`
5. `python -m ml.export_tflite --weights … --parity-data … --quantization float32 …`
6. Package `fusion_v0.tflite` as Android asset (not committed empty).

## Android

**GeoJournal** (subtitle: GeoAI Companion). Source under
`app/src/{main,test,androidTest}/kotlin/…`.

Vertical slice (priority): **camera → Room → optional sync → PostgreSQL → search → delete**.

- **Continue free offline** on login — no backend credentials required
- Private Mode default **on**; cloud sync / enrichment separate opt-ins
- Capture: camera + optional audio/location (permissions + FileProvider)
- `ContextEncoderV1.modalityMask(photo, audio)` — mask is **not** location
- `EdgeMemoryAnalyzer` / `fusion_v0.tflite` (honest unavailable without extractors)
- Train Mode labels before model reveal
- WorkManager outbox when cloud sync + JWT
- **Privacy → Export consented training bronze** (local JSONL+zip for `prepare_fusion_dataset`)
- Optional on-device encoders: package `mobilenet_v3_small.tflite` + `yamnet_meanpool.tflite` via
  `python -m ml.export_encoders_tflite --out-dir app/src/main/assets`

### API base URL (credentials / endpoints)

`BuildConfig.MEMORY_API_BASE_URL` is injected from repo-root `local.properties`
(or defaults to the emulator host alias):

```properties
# local.properties (gitignored) — see secrets/android.local.properties.example
MEMORY_API_BASE_URL=http://10.0.2.2:8000/
```

Release builds require an `https://` base URL.

## Backend

Fail-closed **FastAPI** + SQLAlchemy → **PostgreSQL 16 + pgvector**.

- Env: `backend/.env.example` (`JWT_SECRET` + `GEO_DATABASE_URL` required)
- Schema: `001_memories_pgvector.sql` then **`002_ai_ml_alignment.sql`**
  (E5 semantic **1024-D**, perceptual 128-D, `training_labels`)
- Rules: reject `private_mode` uploads; never invent vibes; enrichment gated;
  no Spark/training in the request path
- Train labels: `POST /api/training/labels` (requires `consent_for_cloud`)

```bash
cp backend/.env.example backend/.env
# set JWT_SECRET + GEO_DATABASE_URL

# optional local demo account (never production):
# GEO_SEED_DEMO_USER=1
# GEO_DEMO_PASSWORD=a-long-local-only-secret

pip install -r backend/requirements-dev.txt
GEO_TEST_MODE=1 JWT_SECRET=test-only-jwt-secret-not-for-production \
  GEO_DATABASE_URL=sqlite+pysqlite:///:memory: \
  python -m pytest -q backend/tests
```

## Public data

- ESC-10: sound pipeline benchmark (CC BY)
- ESC-50 full: noncommercial research only
- FSD50K: start with CC0 for event heads — never as vibe truth

## License / privacy

No raw personal media or model weights in git. HF uploads = deidentified
embeddings + labels + provenance only, private repos, pinned revisions.
