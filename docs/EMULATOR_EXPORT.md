# Emulator export guide (assets + models)

You run exports on the host (pyenv TF / E5 service). The agent and Android emulator
consume the resulting **files** under `app/src/main/assets/` and the **Postgres**
schema. This document is the single checklist.

## Embedding sizes (lock these)

| Space | Dim | Asset / service | Kotlin constant | Postgres |
|-------|-----|-----------------|-----------------|----------|
| Image (MobileNet) | **576** | `mobilenet_v3_small.tflite` | `EmbeddingContract.IMAGE_DIM` | features only |
| Audio (YAMNet) | **1024** | `yamnet_meanpool.tflite` | `AUDIO_DIM` | features only |
| Context | **12** | code | `CONTEXT_DIM` | structured_evidence |
| Mask | **3** | code | `MASK_DIM` | structured_evidence |
| Vibe probs | **7** | `fusion_v0.tflite` out | `VIBE_PROBS_DIM` | structured_evidence |
| Perceptual | **128** | `fusion_v0.tflite` out | `PERCEPTUAL_DIM` | `vector(128)` |
| **Semantic E5** | **1024** | HTTP `127.0.0.1:6100` | `SEMANTIC_DIM` | `memory_semantic_embeddings` |

**Never** treat YAMNet 1024 and E5 1024 as the same space.

E5 model: **`intfloat/e5-large-v2`**  
Prefixes: `query: …` / `passage: …`

---

## 0. One-time host setup

```bash
cd /home/al/Documents/oui/interviews/Android-App-Geolocation
pyenv shell 3.12.13
set -a && source .env && set +a
export GEO_DATABASE_URL="${GEO_DATABASE_URL:-$DATABASE_URL}"

# Postgres already: EffuzionBridgePhoneApp + migrations 001–004
# E5 health:
curl -s http://127.0.0.1:6100/health
# → {"status":"ok","model":"intfloat/e5-large-v2","dim":1024}
```

Apply text 1024 migration if not yet:

```bash
psql "$GEO_DATABASE_URL" -f backend/migrations/004_text_embedding_1024.sql
```

---

## 0b. Optional: sparse MoE capacity (same TFLite I/O)

Dense `fusion_v0` remains the **release baseline**. MoE is opt-in:

```bash
python -m ml.experiments.train_moe_v0 \
  --manifest /secure/fusion_data/manifest.json \
  --weights-out ml/artifacts/fusion-moe-v0.weights.h5 \
  --num-experts 4 --top-k 2 --epochs 40 --seed 42

# Export with same signatures as dense (Kotlin unchanged):
python -m ml.export_tflite \
  --architecture moe \
  --weights ml/artifacts/fusion-moe-v0.weights.h5 \
  --parity-data /secure/fusion_data/eval/parity.npz \
  --out app/src/main/assets/fusion_v0.tflite \
  --savedmodel-dir ml/artifacts/fusion-moe-v0/saved_model \
  --quantization float32 --force
```

Ship as `fusion_v0.tflite` only if held-out macro-F1 **beats** dense on the same sessions.

## 1. Export fusion head (after Train Mode data + train)

```bash
# 1) Labels: app Privacy → "Export consented training bronze"
# 2) Copy bronze JSONL / zip to training machine
# 3) Prepare NPZ + train (see docs/TRAINING_FUSION_V0.md)

python -m ml.prepare_fusion_dataset --input /path/to/bronze --out-dir /secure/fusion_data
python -m ml.train_fusion_v0 \
  --manifest /secure/fusion_data/manifest.json \
  --weights-out ml/artifacts/fusion_v0.weights.h5 \
  --epochs 60 --seed 42

python -m ml.export_tflite \
  --weights ml/artifacts/fusion_v0.weights.h5 \
  --out app/src/main/assets/fusion_v0.tflite \
  --quantization float32
```

Expected asset: **`app/src/main/assets/fusion_v0.tflite`**

---

## 2. Export on-device encoders (MobileNet + YAMNet)

```bash
python -m ml.export_encoders_tflite --out-dir app/src/main/assets
```

Produces (when TF / hub available):

| File | Dim out |
|------|---------|
| `mobilenet_v3_small.tflite` | 576 |
| `yamnet_meanpool.tflite` | 1024 (YAMNet, not E5) |

Without these assets, Kotlin stays **honest Unavailable** for on-device vibe
(`extractorsReady=false`). Capture + Room + sync still work.

---

## 3. E5 semantic (host service → Postgres, not TFLite on phone v0)

E5 stays on the **host** at port **6100** (OpenClaw bridge optional at 6200).

```bash
# Smoke
python -m ml.semantic_e5 --text "café in Lisbon"

# Fill memory_semantic_embeddings after captions exist
python -m backend.scripts.backfill_e5_embeddings
```

Android stores optional cache later in `semanticEmbeddingJson` (1024 floats JSON).
Emulator does **not** need an E5 `.tflite` for the vertical slice.

---

## 4. Point emulator at host backend + E5

| Consumer | URL |
|----------|-----|
| Android app → FastAPI | `MEMORY_API_BASE_URL=http://10.0.2.2:8000/` in **repo-root** `local.properties` |
| FastAPI → Postgres | `GEO_DATABASE_URL` / `DATABASE_URL` in `.env` |
| FastAPI / scripts → E5 | `GEO_E5_BASE_URL=http://127.0.0.1:6100` |

```properties
# local.properties (gitignored)
sdk.dir=/home/YOU/Android/Sdk
MEMORY_API_BASE_URL=http://10.0.2.2:8000/
```

Start API (host):

```bash
set -a && source .env && set +a
export GEO_DATABASE_URL="${GEO_DATABASE_URL:-$DATABASE_URL}"
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

`10.0.2.2` is the emulator’s alias for the host loopback.

---

## 5. Build + install on emulator

```bash
# After assets exist (even empty fusion is OK for compile; runtime stays unavailable)
./gradlew :app:assembleDebug
adb devices
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.geolocation.debug/.MainActivity
```

Pull a bronze export off device (after Train Mode + Privacy export):

```bash
adb shell "run-as com.example.geolocation.debug ls files/training_export"
adb exec-out run-as com.example.geolocation.debug \
  cat files/training_export/geoai_bronze_XXXX.zip > /tmp/bronze.zip
```

---

## 6. What you do vs what the agent does

| You | Agent (when you say so) |
|-----|-------------------------|
| Keep E5 :6100 running | Call `/health` + embeddings |
| Run `export_encoders_tflite` / `export_tflite` if TF ready | Wire assets paths, verify dims |
| Start emulator + `adb install` | `adb logcat`, install if SDK present |
| Train Mode + bronze export on device | Pull zip, run prepare/train scripts |
| Confirm DB password in `.env` | Migrations, backfill, API tests |

After you drop assets or finish an export, tell the agent — it can verify:

```bash
curl -s http://127.0.0.1:6100/health
ls -la app/src/main/assets/
psql "$GEO_DATABASE_URL" -c "\d memory_semantic_embeddings"
adb devices
```

---

## 7. Asset checklist (emulator-ready)

```text
app/src/main/assets/
  fusion_v0.tflite              # optional until trained
  mobilenet_v3_small.tflite     # optional until export_encoders
  yamnet_meanpool.tflite        # optional until export_encoders
```

Large binaries should stay **gitignored**; package for local/release builds only.
