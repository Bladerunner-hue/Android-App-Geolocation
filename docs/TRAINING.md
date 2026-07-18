# How to Train GeoAI MoE (Minimal Data)

**Goal:** functional multimodal vibe model in days, not months, without tens of thousands of labels.

## Why this works with little data

| Lever | What it does |
|--------|----------------|
| Transfer learning | EfficientNet-B0 ImageNet frozen; only fusion + MoE + heads train first |
| LoRA | Low-rank adapters on dense/experts → ~5% trainable params (see smoke: ~200k / 4.6M) |
| Heavy `tf.data` augment | Image flip/brightness, SpecAugment-lite, geo jitter → 10–100× effective data |
| CoT multi-task | Intermediate slot labels give structured signal even with weak tags |
| Manifest class weights | Inverse-sqrt frequency stops majority-class collapse |
| Gradient accumulation | Effective large batch on one GPU (no MirroredStrategy) |

**Target labels:** 100–500 examples per vibe class is enough to beat a rules baseline. Start with synthetic bootstrap → replace with app memories + ESC-50 audio mapping.

## 15-minute path (works offline today)

```bash
cd interviews/Android-App-Geolocation

# 1) Synthetic TFRecords (200 train / 40 val) — class-correlated so learning is real
python -m ml.synthetic_bootstrap

# 2) Train (use --cpu if cuDNN/driver is broken)
CUDA_VISIBLE_DEVICES="" python -m ml.train --cpu --epochs 8 --batch-size 8 --export

# 3) Hybrid serve: MoE → Ollama → Grok → rules
python -m ml.serve_fallback

# 4) Optional: on-device export
python -m ml.export_tflite
```

Smoke-test the model only:

```bash
CUDA_VISIBLE_DEVICES="" python moe_kickstart.py
# → SMOKE OK
```

## Real personal data (still tiny)

1. In the Android app, capture photo + 5–10s ambient audio + lat/lon + optional user vibe tag.
2. Log events as JSONL Bronze (paths + labels), e.g. `ml/data_sample/bronze_events.jsonl`.
3. Gold features:

```bash
python backend/jobs/pyspark_export_gold.py
# or with Spark:  python backend/jobs/pyspark_export_gold.py --spark
```

4. Convert Gold rows → TFRecords (image bytes + int16 log-mel + geo + vibe + cot) with the same schema as `ml/synthetic_bootstrap.py` (`FEATURE_SPEC`, `SCHEMA_VERSION=1`).
5. Point training at your manifests:

```bash
python -m ml.train \
  --train-manifest path/to/manifest_train.json \
  --val-manifest path/to/manifest_val.json \
  --epochs 12 --export
```

### Public bootstrap corpora

- **ESC-50** → map environment classes to vibes (`rain`→serene, `crowd`→social/energetic, `siren`→tense, …).
- **Places365 / your camera roll** subset for vision tints.
- **Synthetic geo** from hour-of-day + lat/lon normalization until trajectory features exist.

## Architecture (train-time)

```
image [B,224,224,3] ──► EfficientNet-B0 (frozen) ──► LoRA proj ─┐
audio log-mel [B,96,64,1] ──► CNN ──► LoRA ─────────────────────┼─► tokens [B,4,H]
geo [B,8] + Fourier ──► LoRA ───────────────────────────────────┘
                              │
                    MoETransformerBlock × N
                    (MHA + top-k SparseMoE)
                              │
              vibe_logits(7)  cot_logits(4×8)  insight_embedding(H)
```

Loss: `L = CE_vibe (smoothed, weighted) + λ · CE_cot`  
Optimizer: AdamW + warmup + cosine; optional mixed_float16; final logits **float32**.

## Serving & fallback (always functional)

```
try:    GeoAIMoE SavedModel / TFLite
except: Ollama (local llama3.x, private)
except: Grok API (XAI_API_KEY)
else:   deterministic rules caption
```

`ml/serve_fallback.py` implements this chain. Wire into FastAPI `/vision/analyze` when you extend the backend.

## PySpark rules (scale path)

- **No TensorFlow in Spark ETL** — only in executor-side TFRecord writers if needed.
- Never `collect()` images/audio to the driver.
- Persist Silver after expensive windows; Adaptive Query Execution on.
- Manifest next to shards: row counts, SHA-256, class counts → class weights.

## RLAIF later (optional polish)

When you have preference pairs (Grok/Ollama synthetic CoT “better/worse” journal captions):

- Gate KTO/DPO via policy flags (tensorflow-talex `rlhf_gates` pattern).
- Ramp preference loss weight 0 → 0.4 → decay; never train RL-only from day one.

## Checklist

- [ ] `python moe_kickstart.py` → SMOKE OK  
- [ ] Synthetic bootstrap + 1 epoch train completes  
- [ ] Val accuracy rises above chance (~0.14 for 7 classes) on correlated synth  
- [ ] SavedModel export loads in `serve_fallback`  
- [ ] Ollama fallback returns JSON when MoE weights missing  
- [ ] TFLite file size reasonable for APK assets  
- [ ] ADR-002 reviewed  

## File map

| Path | Role |
|------|------|
| `moe_kickstart.py` | Model, losses, grad-accum trainer |
| `ml/synthetic_bootstrap.py` | Day-1 TFRecords + manifest |
| `ml/tf_data_pipeline.py` | Parse + augment + prefetch |
| `ml/train.py` | Training loop + SavedModel export |
| `ml/export_tflite.py` | INT8 edge export |
| `ml/serve_fallback.py` | MoE / Ollama / Grok / rules |
| `backend/jobs/pyspark_export_gold.py` | Medallion Gold + weights |
| `docs/adr/ADR-002-ai-layer.md` | Architecture decision |
