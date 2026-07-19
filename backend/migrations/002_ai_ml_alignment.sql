-- 002_ai_ml_alignment.sql
-- Apply after 001. Supports Kotlin offline-first + fusion_v0 / future MoE data plane.
--
--   psql "$GEO_DATABASE_URL" -f backend/migrations/001_memories_pgvector.sql
--   psql "$GEO_DATABASE_URL" -f backend/migrations/002_ai_ml_alignment.sql
--
-- Contracts (see docs/CONFIRMATION.md):
--   perceptual_embedding  vector(128)  — fusion_v0 perceptual head
--   text_embedding        vector(768)  — multilingual semantic encoder (not fusion)
--   insight_embedding     vector(128)  — optional auxiliary insight space
--   Vibes only from human Train Mode labels for training; analyze never invents vibes.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 1. Upgrade text embedding to real semantic size (768)
-- Nulls any legacy toy hash (64-D). Re-embed with a registered encoder after cutover.
-- ---------------------------------------------------------------------------
ALTER TABLE memories
    ALTER COLUMN text_embedding TYPE vector(768)
    USING NULL;

-- ---------------------------------------------------------------------------
-- 2. AI/ML columns on memories
-- ---------------------------------------------------------------------------
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS insight_embedding vector(128);

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS model_version VARCHAR(64);

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS analysis_source VARCHAR(32) DEFAULT 'unavailable';
    -- on_device | server_fusion | rules | unavailable

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS structured_evidence JSONB;
    -- Production evidence: vibe_probs[7], modality_mask[3], context12, etc.

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS enrichment_requested BOOLEAN NOT NULL DEFAULT FALSE;

-- ---------------------------------------------------------------------------
-- 3. Train Mode labels (mirrors Kotlin MemoryTrainingLabelEntity)
-- Corrections append new rows; corrects_label_id points at prior label.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_labels (
    id                      UUID PRIMARY KEY,
    memory_id               VARCHAR(64) NOT NULL,
    user_id                 INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id              VARCHAR(64) NOT NULL,
    primary_vibe            VARCHAR(32) NOT NULL,
    secondary_vibes         JSONB NOT NULL DEFAULT '[]',
    valence                 SMALLINT,
    arousal                 SMALLINT,
    label_confidence        SMALLINT NOT NULL,
    label_source            VARCHAR(32) NOT NULL DEFAULT 'human_self',
    utc_offset_minutes      INTEGER NOT NULL,
    location_accuracy_m     REAL,
    consent_for_training    BOOLEAN NOT NULL DEFAULT FALSE,
    consent_for_cloud       BOOLEAN NOT NULL DEFAULT FALSE,
    labelled_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    corrects_label_id       UUID,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_training_valence CHECK (valence IS NULL OR (valence >= -2 AND valence <= 2)),
    CONSTRAINT chk_training_arousal CHECK (arousal IS NULL OR (arousal >= 1 AND arousal <= 5)),
    CONSTRAINT chk_training_confidence CHECK (label_confidence >= 1 AND label_confidence <= 3)
);

CREATE INDEX IF NOT EXISTS idx_training_labels_user ON training_labels(user_id);
CREATE INDEX IF NOT EXISTS idx_training_labels_memory ON training_labels(memory_id);
CREATE INDEX IF NOT EXISTS idx_training_labels_session ON training_labels(session_id);
CREATE INDEX IF NOT EXISTS idx_training_labels_consent ON training_labels(consent_for_training)
    WHERE consent_for_training = TRUE;

-- Optional ANN indexes once volume supports IVF lists / HNSW build
-- CREATE INDEX IF NOT EXISTS idx_memories_perceptual_ivfflat ON memories
--     USING ivfflat (perceptual_embedding vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX IF NOT EXISTS idx_memories_text_ivfflat ON memories
--     USING ivfflat (text_embedding vector_cosine_ops) WITH (lists = 100);
