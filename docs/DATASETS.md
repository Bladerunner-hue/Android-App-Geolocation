# Datasets and licensing

## Personal Train Mode (required for vibe)

Public sound/image corpora do **not** contain truthful labels for the personal
seven-class vibe taxonomy. Rain is objectively rain; it is not objectively serene.

Train Mode stores:

| Field | Notes |
|-------|--------|
| primary_vibe | One of 7 labels |
| secondary vibes | Optional |
| valence / arousal | Continuous axes |
| label_confidence | 1–3 |
| label_source | `human_self`, `human_reviewed`, `human_quick` |
| session_id | Rotates after ~60 min inactivity |
| consent_for_training | Separate from cloud |
| consent_for_cloud | Separate from training |
| corrections | Append-only, never silent overwrite |

**Ask for the label before showing the model prediction.**

Suggested volumes (heuristics):

- Wiring: ~30/class (210)
- Internal v0: 100–150/class
- Strong personalization: 300–500/class across many sessions

## Frozen encoders (use immediately)

| Encoder | Output | License / source |
|---------|--------|------------------|
| MobileNetV3Small ImageNet | 576-D | Keras applications |
| YAMNet (TF-Hub) | 1024-D mean frames | Apache-2.0 model |

## Public sound (auxiliary events only)

| Dataset | Use | License caution |
|---------|-----|-----------------|
| ESC-10 | Pipeline benchmark | CC BY (subset of ESC-50) |
| ESC-50 full | Research only | **CC BY-NC** — not commercial weights |
| FSD50K CC0 | Sound-event bootstrap | Preferred |
| FSD50K CC BY | More events | Attribution + legal review |
| Image emotion sets | Skip for v0 | Rights often restricted |

Never fabricate vibe labels from ESC/FSD classes. Use `ml/sound_bootstrap.py`.

## Semantic text search (separate)

`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` → 768-D Apache-2.0.
Store in `semantic_embedding VECTOR(768)`. Do **not** pad the 128-D perceptual
vector into the text column.

## Hugging Face (private)

Recommended private repos (example):

- `CenturyGhost/geojournal-personal-features-v1` — deidentified embeddings only
- `CenturyGhost/geojournal-sound-bootstrap-v1`
- `CenturyGhost/geojournal-vibe-fusion-v0` — model artifacts

Do **not** upload raw photos, audio, notes, exact GPS, Room DBs, or tokens.
Pin dataset/model revisions by commit SHA.
