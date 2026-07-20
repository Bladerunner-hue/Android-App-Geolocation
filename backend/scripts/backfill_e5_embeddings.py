"""Backfill memory_semantic_embeddings via direct E5 HTTP service (port 6100).

Uses GEO_DATABASE_URL (or DATABASE_URL) and GEO_E5_BASE_URL.

  set -a && source .env && set +a
  export GEO_DATABASE_URL="$DATABASE_URL"
  export GEO_E5_BASE_URL=http://127.0.0.1:6100
  export GEO_SEMANTIC_MODEL=intfloat/e5-large-v2
  python -m backend.scripts.backfill_e5_embeddings
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = os.environ.get("GEO_SEMANTIC_MODEL", "intfloat/e5-large-v2")
E5_BASE = os.environ.get("GEO_E5_BASE_URL", "http://127.0.0.1:6100")


def main() -> None:
    import psycopg2
    from psycopg2.extras import execute_values

    from ml.semantic_e5 import E5Embedder

    url = os.environ.get("GEO_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url or url.startswith("sqlite"):
        raise SystemExit("Set GEO_DATABASE_URL or DATABASE_URL to PostgreSQL DSN.")

    print(f"E5 service {E5_BASE} model={MODEL_ID}")
    embedder = E5Embedder(base_url=E5_BASE, model_id=MODEL_ID)
    model_id = embedder.model_id
    if embedder.dim != 1024:
        raise SystemExit(f"dim={embedder.dim}, need 1024 for memory_semantic_embeddings")

    conn = psycopg2.connect(url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id, COALESCE(m.caption, ''), m.vibe_label
                FROM memories m
                LEFT JOIN memory_semantic_embeddings e
                  ON e.memory_id = m.id AND e.model_id = %s
                WHERE e.memory_id IS NULL
                  AND (
                    (m.caption IS NOT NULL AND length(trim(m.caption)) > 0)
                    OR m.vibe_label IS NOT NULL
                  )
                ORDER BY m.id
                LIMIT 500
                """,
                (model_id,),
            )
            rows = cur.fetchall()
            if not rows:
                print("Nothing to backfill.")
                return
            texts = []
            ids = []
            for mid, cap, vibe in rows:
                ids.append(mid)
                blob = " ".join(x for x in [cap, vibe or ""] if x).strip()
                texts.append(blob or "memory")
            print(f"Embedding {len(texts)} memories via HTTP …")
            mat = embedder.embed_passages(texts)
            values = [
                (int(mid), model_id, "http-6100", mat[i].tolist())
                for i, mid in enumerate(ids)
            ]
            execute_values(
                cur,
                """
                INSERT INTO memory_semantic_embeddings
                    (memory_id, model_id, model_revision, embedding)
                VALUES %s
                ON CONFLICT (memory_id, model_id) DO UPDATE
                  SET embedding = EXCLUDED.embedding,
                      model_revision = EXCLUDED.model_revision
                """,
                values,
                template="(%s, %s, %s, %s::vector)",
            )
        conn.commit()
        print(f"Upserted {len(values)} semantic embeddings (model_id={model_id}).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
