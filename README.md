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

Docs: [docs/TRAINING_FUSION_V0.md](docs/TRAINING_FUSION_V0.md) ·
[docs/DATASETS.md](docs/DATASETS.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

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

- Private Mode default **on**
- Capture → Room first
- Train Mode Room table `memory_training_labels`
- `ContextEncoderV1` / `FusionV0Interpreter` for on-device path
- WorkManager sync only when explicitly enabled

## Backend

Fail-closed FastAPI (`JWT_SECRET` + `GEO_DATABASE_URL` required). See
`backend/.env.example` and `backend/migrations/001_memories_pgvector.sql`.

```bash
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
