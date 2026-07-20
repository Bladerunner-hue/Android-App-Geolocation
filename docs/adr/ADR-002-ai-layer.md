# ADR-002: Multimodal On-Device + Hybrid AI Layer

**Status:** Accepted  
**Date:** 2026-07-18  
**Decision Makers:** Development Team  
**Technical Story:** Add edge multimodal MoE (vision + ambient audio + geo) with CoT heads, TFServing/TFLite, and Ollama/Grok fallback without requiring large labelled corpora.

---

## Context

The modernized Kotlin/Compose geolocation app needs a portfolio-grade AI Memory Journal. Constraints:

- Personal data volumes are small (hundreds of memories, not millions).
- Privacy-first: pure on-device mode must work offline.
- Must demonstrate 2026 AI engineering: sparse MoE, CoT supervision, KV-cache-ready inference, hybrid routing, PySpark medallion → TFRecord MLOps.

## Decision

1. **Model (experimental)**: Keras `GeoAIMoE` (`ml/experiments/moe_kickstart.py`) — EfficientNet-B0 (frozen) + log-mel CNN + geo Fourier features → pre-norm MultiHeadAttention + sparse top-k MoE FFN → multi-task heads. **Production path is dense fusion_v0 (ADR-003).**
2. **Small-data recipe**: transfer learning + LoRA on dense/experts + heavy `tf.data` augmentation + manifest inverse-sqrt class weights + gradient accumulation + mixed precision.
3. **Data plane**: Bronze → Silver → Gold (PySpark preferred; pure-Python stub for day-1) → TFRecords + `manifest.json` (schema version, shards, SHA-256, class counts). Synthetic bootstrap for CI and first train.
4. **Serving**: SavedModel (backend) + INT8 TFLite (Android). Fallback chain: MoE → Ollama → Grok API → deterministic rules.
5. **Sound analysis**: first-class ambient 5–10s capture → log-mel → vibe fusion (not an afterthought).

## Consequences

- Functional model in days with ~100–500 examples/class (or synthetic bootstrap).
- True sparse GPU kernels deferred; top-k weighted dense experts OK at `num_experts≤8`.
- RLAIF (KTO/DPO) gated later via policy flags — not required for v1 accuracy.

## Related

- `ml/experiments/moe_kickstart.py`, `ml/experiments/train_moe_legacy.py`, `ml/serve_fallback.py`
- `backend/jobs/pyspark_export_gold.py`
- ADR-003 (hybrid inference & privacy) — planned
