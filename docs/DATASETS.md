# Datasets (production-first)

**AI/ML focus locked.** Release weights come from **human Train Mode labels with
`consent_for_training=true` only**. Public corpora and LLM synthesis are bootstrap
or aux — never vibe truth.

## 1. Production data plane (only path that ships weights)

```text
Kotlin capture (photo + 10s 16 kHz WAV + geo + Train Mode human label)
  → Room + structuredEvidenceJson + perceptual (when extractors ready)
  → TrainingBronzeExporter (consent_for_training=true only)
  → backend/jobs/pyspark_export_gold.py   # Bronze → Silver → Gold (pure Spark, no TF)
  → ml/prepare_fusion_dataset.py         # Gold → NPZ/TFRecord + manifest
  → ml/validate_fusion_dataset.py
  → ml/train_fusion_v0.py                # dense fusion_v0 only
  → ml/export_tflite.py                  # assets/fusion_v0.tflite
```

| Contract | Value |
|----------|--------|
| Inputs | `image_embedding[576]`, `audio_embedding[1024]`, `context[12]`, `modality_mask[3]` |
| Outputs | `vibe_probabilities[7]`, `perceptual_embedding` L2 `[128]` |
| Labels | Human Train Mode only — never invent 7-class vibes from ESC/Places |
| Schema | single `SCHEMA_VERSION` in `ml/config.py` |
| Manifest | row counts, class counts, SHA-256, class weights (sqrt-inverse) |
| Consent | backend rejects private uploads; train sees consented rows only |

### Train Mode fields (required for vibe)

| Field | Notes |
|-------|--------|
| primary_vibe | One of 7 labels |
| secondary vibes | Optional |
| valence / arousal | Continuous axes (future multi-task) |
| label_confidence | 1–3 |
| label_source | `human_self`, `human_reviewed`, `human_quick` |
| session_id | Rotates after ~60 min inactivity |
| consent_for_training | Separate from cloud |
| consent_for_cloud | Separate from training |
| corrections | Append-only, never silent overwrite |

**Ask for the label before showing the model prediction.**

Suggested volumes:

| Stage | Heuristic |
|-------|-----------|
| Wiring smoke | ~30/class (~210) via `make_wiring_fixture` |
| Internal v0 | 100–150/class |
| Strong personalization | 300–500/class across many sessions |

## 2. Frozen encoders (use immediately; never retrain)

| Encoder | Output | License / source |
|---------|--------|------------------|
| MobileNetV3Small ImageNet | 576-D | Keras applications |
| YAMNet (TF-Hub) | 1024-D mean frames | Apache-2.0 model |

On-device export: `ml/export_tflite.py` (+ encoder export when present).

## 3. Safe public sources (feature bootstrap / masked aux only)

| Dataset | Use | License caution |
|---------|-----|-----------------|
| ESC-10 | Pipeline / sound-event benchmark | CC BY (ESC-50 subset) |
| ESC-50 full | Research only | **CC BY-NC** — not commercial weights |
| FSD50K CC0 | Sound-event bootstrap | Preferred |
| FSD50K CC BY | More events | Attribution + legal review |
| Places365 small / OpenImages places | Vision feature bootstrap | Map carefully; **no vibe labels** |
| Synthetic geo/time | Fourier hour + lat/lon norm | Until real trajectories exist |

**Never** force a 7-vibe label from environment classes. Helpers live under
`ml/experiments/sound_bootstrap.py` (demoted; not on the release import path).

## 4. Semantic text (separate column — not fusion_v0)

| Encoder | Dim | Notes |
|---------|-----|--------|
| E5 large (`ml/semantic_e5.py` → HTTP :6100) | **1024** | Production text search; `VECTOR(1024)` |
| fusion perceptual | 128 | L2-normalized; **do not pad into text column** |

On-device encoder export (when TF ready): `python -m ml.export_encoders_tflite --out-dir app/src/main/assets`.

## 5. AI-assisted data (allowed, gated)

| Allowed | Forbidden |
|---------|-----------|
| Grok / Ollama synthetic **CoT traces** or preference pairs for later RLAIF (caption quality) | LLM invents primary 7-class vibe for release training |
| Weak multi-label tags / hard-negatives → **human review in Train Mode** | Silent overwrite of human labels |
| RLAIF only after `PolicyStageStatus` + provenance allow; ramp 0 → 0.3 | RLAIF on weak labels alone |

## 6. Repo-lean data artifacts

| Keep in git | Ignore / external |
|-------------|-------------------|
| `ml/data_sample/gold/` manifests + tiny JSONL | `ml/data_sample/tfrecords/*.tfrecord` |
| Wiring fixture manifests (generated, gitignored blobs OK) | `ml/checkpoints/`, `*.weights.h5`, `*.tflite` |
| Schema + class lists in `ml/config.py` | Personal media, exact GPS dumps, Room DBs |

## 7. marimo (exploration only)

```bash
# optional: pip install marimo polars
marimo edit ml/notebooks/validate_fusion_live.py
```

Use for class balance, missing-modality matrix, validate pass/fail, and
copy-paste prepare/train CLI. **Never** embed the production
`train_fusion_v0` loop inside marimo for release runs.

## 8. Hugging Face (private only)

Example private repos (deidentified embeddings + labels + provenance):

- `…/geojournal-personal-features-v1`
- `…/geojournal-sound-bootstrap-v1`
- `…/geojournal-vibe-fusion-v0` — model artifacts

Do **not** upload raw photos, audio, notes, exact GPS, Room DBs, or tokens.
Pin revisions by commit SHA.

## 9. What not to train on

- Synthetic vibe labels from ESC/FSD/Places class names
- Private Mode / non-consented rows
- Experimental MoE TFRecords from `ml/experiments/synthetic_bootstrap.py` for release weights
- Random i.i.d. photo splits (use session-grouped temporal splits)
