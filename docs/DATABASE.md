# Database: PostgreSQL 16 + pgvector + PostGIS

## Host packages (Ubuntu / PGDG)

```bash
# PGDG (if not already configured)
# deb [signed-by=…] http://apt.postgresql.org/pub/repos/apt jammy-pgdg main

sudo apt-get install -y \
  postgresql-16 \
  postgresql-16-pgvector \
  postgresql-16-postgis-3 \
  postgresql-16-postgis-3-scripts
```

Verify:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION postgis;
SELECT extname, extversion FROM pg_extension;
SELECT PostGIS_Version();
```

## Bootstrap schema

```bash
./scripts/init_geojournal_db.sh
# applies 001 → 002 → 003
```

| Migration | Role |
|-----------|------|
| `001_memories_pgvector.sql` | users, memories, vector |
| `002_ai_ml_alignment.sql` | training_labels, AI columns, text 768 placeholder col |
| `003_postgis_e5_semantic.sql` | `geography(Point,4326)`, accuracy/scope, **semantic 1024** + perceptual side tables |

## Embedding spaces (do not mix)

| Space | Dim | Where | Model |
|-------|-----|--------|--------|
| fusion_v0 perceptual | 128 | `memories.perceptual_embedding` or `memory_perceptual_embeddings` | on-device fusion |
| E5 semantic text | **1024** | `memory_semantic_embeddings` | `intfloat/multilingual-e5-large-instruct` |
| Legacy hash / stub | 768 | `memories.text_embedding` | **retire** once E5 backfill runs |

E5-large (and instruct) is **1024-D**, not 768. Use the side table from 003.

## Backfill captions with E5

```bash
pyenv shell 3.12.13
pip install sentence-transformers
export GEO_DATABASE_URL=postgresql://geojournal:geojournal_dev_change_me@127.0.0.1:5432/geojournal
python -m backend.scripts.backfill_e5_embeddings
# optional: GEO_SEMANTIC_MODEL=intfloat/multilingual-e5-large-instruct
```

Query sketch (spatial + semantic):

```sql
SELECT m.id, m.caption,
       1 - (e.embedding <=> $query_vec) AS score
FROM memory_semantic_embeddings e
JOIN memories m ON m.id = e.memory_id
WHERE m.user_id = $uid
  AND e.model_id = 'intfloat/multilingual-e5-large-instruct'
  AND ST_DWithin(
        m.location,
        ST_SetSRID(ST_MakePoint($lon, $lat), 4326)::geography,
        $radius_m
      )
ORDER BY e.embedding <=> $query_vec
LIMIT 20;
```

## Dev DSN

```
GEO_DATABASE_URL=postgresql://geojournal:geojournal_dev_change_me@127.0.0.1:5432/geojournal
```

Change the password for anything non-local.
