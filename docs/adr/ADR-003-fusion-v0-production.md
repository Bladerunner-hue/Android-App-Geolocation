# ADR-003: Dense fusion_v0 as production vibe model

**Status:** Accepted  
**Date:** 2026-07-19  

## Decision

Ship **dense fusion_v0** (frozen MobileNetV3Small + YAMNet + context12 + mask → ~280k-param head) as the Android production path. Keep sparse MoE / raw-media transformers experimental only.

## Context

- Personal vibe labels are subjective; public sound datasets must not invent vibe classes.
- Edge devices need small, parity-testable TFLite graphs without fake CoT/KV.
- Feature extraction is the expensive step; the head is cheap on A5000.

## Consequences

- Train Mode + session-grouped NPZ pipeline required before real accuracy claims.
- Semantic search remains a separate 768-D text encoder.
- MoE revisited only if dense capacity fails on much larger personal data.
