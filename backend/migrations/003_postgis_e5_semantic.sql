-- 003_postgis_e5_semantic.sql
-- PostGIS geography + versioned semantic embeddings (E5-large family).
--
-- Prerequisites on the host:
--   sudo apt-get install -y postgresql-16-postgis-3 postgresql-16-postgis-3-scripts
--   (pgvector already: postgresql-16-pgvector)
--
-- Apply (after 001 + 002):
--   psql "$GEO_DATABASE_URL" -f backend/migrations/001_memories_pgvector.sql
--   psql "$GEO_DATABASE_URL" -f backend/migrations/002_ai_ml_alignment.sql
--   psql "$GEO_DATABASE_URL" -f backend/migrations/003_postgis_e5_semantic.sql
--
-- Embedding contracts:
--   fusion_v0 perceptual  → vector(128)  on memories.perceptual_embedding (or side table)
--   E5 multilingual large → vector(1024) in memory_semantic_embeddings  (NOT fusion space)
--
-- Model id used by the offline/API encoder helper:
--   intfloat/multilingual-e5-large-instruct   (1024-D; use "query: "/"passage: " prefixes)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- Location as first-class geography (WGS84), meters-friendly ST_DWithin
-- ---------------------------------------------------------------------------
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS location geography(Point, 4326);

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS location_accuracy_m real;

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS location_scope text NOT NULL DEFAULT 'none';
    -- none | exact | coarse  (privacy; EXIF strip is app-layer)

-- Backfill from existing lat/lon when present
UPDATE memories
SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
WHERE location IS NULL
  AND latitude IS NOT NULL
  AND longitude IS NOT NULL;

-- Keep lat/lon columns for API simplicity; trigger keeps geography in sync
CREATE OR REPLACE FUNCTION memories_sync_location()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF NEW.latitude IS NOT NULL AND NEW.longitude IS NOT NULL THEN
    NEW.location := ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326)::geography;
    IF NEW.location_scope = 'none' THEN
      NEW.location_scope := 'exact';
    END IF;
  ELSE
    NEW.location := NULL;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_memories_sync_location ON memories;
CREATE TRIGGER trg_memories_sync_location
    BEFORE INSERT OR UPDATE OF latitude, longitude
    ON memories
    FOR EACH ROW
    EXECUTE FUNCTION memories_sync_location();

CREATE INDEX IF NOT EXISTS idx_memories_location_gist
    ON memories USING gist (location);

CREATE INDEX IF NOT EXISTS idx_memories_user_captured
    ON memories (user_id, captured_at DESC);

-- ---------------------------------------------------------------------------
-- Versioned semantic embeddings (E5 1024-D)
-- Do not store E5 vectors in fusion_v0 perceptual columns.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_semantic_embeddings (
    memory_id       integer NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_id        text NOT NULL,
    model_revision  text NOT NULL DEFAULT '',
    embedding       vector(1024) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_memories_semantic_hnsw
    ON memory_semantic_embeddings
    USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE memory_semantic_embeddings IS
  'Text/caption semantic space. Default model: intfloat/multilingual-e5-large-instruct (1024-D).';

-- Optional: side table for perceptual if you want multi-revision fusion heads
CREATE TABLE IF NOT EXISTS memory_perceptual_embeddings (
    memory_id       integer NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_id        text NOT NULL DEFAULT 'fusion_v0',
    model_revision  text NOT NULL DEFAULT 'r1',
    embedding       vector(128) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_memories_perceptual_hnsw
    ON memory_perceptual_embeddings
    USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Example spatial + semantic query (application uses parameters):
--
-- SELECT m.id, m.caption, m.captured_at,
--        1 - (e.embedding <=> :query_vec) AS semantic_score
-- FROM memory_semantic_embeddings e
-- JOIN memories m ON m.id = e.memory_id
-- WHERE m.user_id = :user_id
--   AND e.model_id = 'intfloat/multilingual-e5-large-instruct'
--   AND m.captured_at >= :from_time AND m.captured_at < :to_time
--   AND ST_DWithin(
--         m.location,
--         ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
--         :radius_m
--       )
-- ORDER BY e.embedding <=> :query_vec
-- LIMIT :limit;
