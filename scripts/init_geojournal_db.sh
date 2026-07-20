#!/usr/bin/env bash
# Bootstrap local PostgreSQL 16 + pgvector + PostGIS for GeoJournal.
# Safe to re-run (IF NOT EXISTS / CREATE EXTENSION).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_NAME="${GEO_DB_NAME:-geojournal}"
DB_USER="${GEO_DB_USER:-geojournal}"
DB_PASS="${GEO_DB_PASSWORD:-geojournal_dev_change_me}"
HOST="${GEO_DB_HOST:-127.0.0.1}"
PORT="${GEO_DB_PORT:-5432}"

echo "==> Ensuring role + database ($DB_NAME)"
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}';
  END IF;
END\$\$;
SQL

if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
fi

echo "==> Extensions (vector + postgis)"
sudo -u postgres psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<SQL
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;
GRANT ALL ON SCHEMA public TO ${DB_USER};
ALTER DATABASE ${DB_NAME} OWNER TO ${DB_USER};
SQL

echo "==> Migrations 001 → 003"
for f in \
  "${ROOT}/backend/migrations/001_memories_pgvector.sql" \
  "${ROOT}/backend/migrations/002_ai_ml_alignment.sql" \
  "${ROOT}/backend/migrations/003_postgis_e5_semantic.sql"
do
  echo "    applying $(basename "$f")"
  sudo -u postgres psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -f "$f"
done

sudo -u postgres psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<SQL
GRANT ALL ON ALL TABLES IN SCHEMA public TO ${DB_USER};
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO ${DB_USER};
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ${DB_USER};
SQL

echo "==> Done"
echo "GEO_DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@${HOST}:${PORT}/${DB_NAME}"
echo "PostGIS: SELECT PostGIS_Version();"
echo "pgvector: SELECT extversion FROM pg_extension WHERE extname='vector';"
echo "E5 semantic table: memory_semantic_embeddings (vector 1024)"
