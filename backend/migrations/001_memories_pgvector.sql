-- GeoJournal PostgreSQL 16 + pgvector migration (baseline)
-- Apply:
--   psql "$GEO_DATABASE_URL" -f backend/migrations/001_memories_pgvector.sql
--   psql "$GEO_DATABASE_URL" -f backend/migrations/002_ai_ml_alignment.sql
--
-- Note: 002 upgrades text_embedding to vector(768) and adds training_labels + AI columns.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    client_uuid VARCHAR(64) NOT NULL,
    caption TEXT,
    vibe_label VARCHAR(32),
    vibe_confidence DOUBLE PRECISION,
    analysis_status VARCHAR(32) NOT NULL DEFAULT 'unavailable',
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    private_mode BOOLEAN NOT NULL DEFAULT FALSE,
    photo_path VARCHAR(512),
    audio_path VARCHAR(512),
    -- Separate indexes: perceptual (image/audio) vs text (query encoder)
    perceptual_embedding vector(128),
    text_embedding vector(64),
    evidence_json JSONB,
    content_sha256 VARCHAR(64),
    CONSTRAINT uq_memory_user_client_uuid UNIQUE (user_id, client_uuid)
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_captured_at ON memories(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_vibe ON memories(vibe_label);

-- Optional ANN indexes when volume grows (requires enough rows for lists)
-- CREATE INDEX IF NOT EXISTS idx_memories_text_ivfflat
--   ON memories USING ivfflat (text_embedding vector_cosine_ops) WITH (lists = 100);
