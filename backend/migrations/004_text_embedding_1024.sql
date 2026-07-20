-- 004_text_embedding_1024.sql
-- Align memories.text_embedding width with E5 e5-large-v2 (1024-D).
-- Canonical semantic store remains memory_semantic_embeddings (vector 1024).
--
--   psql "$GEO_DATABASE_URL" -f backend/migrations/004_text_embedding_1024.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- Drop incompatible legacy vectors (64/768 placeholders); re-embed via E5 service.
ALTER TABLE memories
    ALTER COLUMN text_embedding TYPE vector(1024)
    USING NULL;

COMMENT ON COLUMN memories.text_embedding IS
  'Optional inline 1024-D vector; prefer memory_semantic_embeddings with model_id=intfloat/e5-large-v2';
