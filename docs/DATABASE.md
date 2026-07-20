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
| E5 semantic text | **1024** | `memory_semantic_embeddings` | `intfloat/e5-large-v2` (HTTP :6100) |
| Legacy hash / stub | 768 | `memories.text_embedding` | **retire** once E5 backfill runs |

E5-large (and instruct) is **1024-D**, not 768. Use the side table from 003.

## Direct E5 HTTP service (preferred)

Live local stack:

| Port | Role |
|------|------|
| **6100** | Direct E5: `intfloat/e5-large-v2`, dim **1024** |
| **6200** | OpenClaw memory bridge → calls 6100 |

```bash
curl -s http://127.0.0.1:6100/health
# {"status":"ok","model":"intfloat/e5-large-v2","dim":1024}

# OpenAI-compatible
curl -s http://127.0.0.1:6100/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"intfloat/e5-large-v2","input":"query: café in Lisbon"}'
```

Python client uses that service (no local sentence-transformers required):

```bash
pyenv shell 3.12.13
set -a && source .env && set +a
export GEO_DATABASE_URL="${GEO_DATABASE_URL:-$DATABASE_URL}"
export GEO_E5_BASE_URL=http://127.0.0.1:6100
export GEO_SEMANTIC_MODEL=intfloat/e5-large-v2

python -m ml.semantic_e5 --text "café in Lisbon"
python -m backend.scripts.backfill_e5_embeddings
```

E5 prefixes: `query: …` for search, `passage: …` for captions stored in DB.

## Database from repo `.env`

Create role/DB from your project `.env` (`DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DATABASE_URL`):

```bash
# already applied on this host for EffuzionBridgePhoneApp + migrations 001–003
# re-run:
set -a && source .env && set +a
# use scripts/init_geojournal_db.sh with GEO_DB_* overrides, or create via sudo -u postgres
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
