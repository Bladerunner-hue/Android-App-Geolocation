"""Telemetry ingestion endpoint for Android edge telemetry.

Receives JSONL (NDJSON) event batches from TelemetrySyncWorker.
Stores raw events in the media root under telemetry/ for downstream
PySpark / Bronze pipeline consumption.

No auth — telemetry is anonymous per install_id.
Rate-limited by file size (max 1 MB per batch).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from backend.app.config import Settings, get_settings

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])

MAX_BATCH_BYTES = 1_024 * 1024  # 1 MB per batch


def _telemetry_root(settings: Settings) -> Path:
    root = settings.media_root / "telemetry"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _hour_prefix() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H")


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_telemetry(
    request: Request,
    x_telemetry_source: Optional[str] = Header(None, alias="X-Telemetry-Source"),
    x_install_id: Optional[str] = Header(None, alias="X-Install-Id"),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Accept a batch of NDJSON telemetry events from Android edge.

    Events are written to ``{GEO_MEDIA_ROOT}/telemetry/{hour}/``
    with a random filename for idempotency.  No database writes —
    this is a raw ingestion pipe for the PySpark Bronze layer.
    """
    body = await request.body()
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="Empty body")
    if len(body) > MAX_BATCH_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch exceeds {MAX_BATCH_BYTES} bytes",
        )

    # Quick sanity: every line must start with '{'
    text = body.decode("utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            raise HTTPException(
                status_code=400,
                detail=f"Line {i} is not valid JSON (must start with '{{')",
            )

    source = (x_telemetry_source or "unknown").strip()
    install = (x_install_id or "unknown").strip()

    root = _telemetry_root(settings)
    hour_dir = root / _hour_prefix()
    hour_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent filename: hash of body + install id
    digest = hashlib.sha256(body).hexdigest()[:16]
    fname = f"{source}_{install}_{digest}_{uuid.uuid4().hex[:8]}.jsonl"
    dest = hour_dir / fname

    dest.write_bytes(body)

    return {
        "status": "accepted",
        "bytes": len(body),
        "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
        "file": str(dest.relative_to(root)),
    }


@router.get("/health")
def telemetry_health(settings: Settings = Depends(get_settings)) -> dict:
    root = _telemetry_root(settings)
    files = list(root.rglob("*.jsonl"))
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    return {
        "status": "ok",
        "endpoint": "/api/telemetry/ingest",
        "stored_files": len(files),
        "stored_bytes": total_bytes,
        "max_batch_bytes": MAX_BATCH_BYTES,
    }
