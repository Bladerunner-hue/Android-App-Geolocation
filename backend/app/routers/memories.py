"""Memory upload / search / get."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.config import Settings, get_settings
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import AnalyzeMeta, MemoryResponse, MemorySearchResponse
from backend.app.services.memory_service import MemoryService

router = APIRouter(prefix="/api/memories", tags=["memories"])


def _parse_meta(
    client_uuid: str,
    private_mode: bool,
    caption: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    captured_at: Optional[str],
    on_device_vibe: Optional[str],
    on_device_confidence: Optional[float],
    on_device_probs: Optional[str],
    request_enrichment: bool,
) -> AnalyzeMeta:
    probs = None
    if on_device_probs:
        probs = json.loads(on_device_probs)
    ts = None
    if captured_at:
        ts = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).replace(tzinfo=None)
    return AnalyzeMeta(
        client_uuid=client_uuid,
        private_mode=private_mode,
        caption=caption,
        latitude=latitude,
        longitude=longitude,
        captured_at=ts,
        on_device_vibe=on_device_vibe,
        on_device_confidence=on_device_confidence,
        on_device_probs=probs,
        request_enrichment=request_enrichment,
    )


@router.post("/analyze", response_model=MemoryResponse)
async def analyze_memory(
    client_uuid: str = Form(...),
    private_mode: bool = Form(False),
    caption: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    captured_at: Optional[str] = Form(None),
    on_device_vibe: Optional[str] = Form(None),
    on_device_confidence: Optional[float] = Form(None),
    on_device_probs: Optional[str] = Form(None),
    request_enrichment: bool = Form(False),
    photo: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> MemoryResponse:
    meta = _parse_meta(
        client_uuid,
        private_mode,
        caption,
        latitude,
        longitude,
        captured_at,
        on_device_vibe,
        on_device_confidence,
        on_device_probs,
        request_enrichment,
    )
    svc = MemoryService(db, settings)
    mem, _created = await svc.analyze_upload(user, meta, photo, audio)
    return mem


@router.get("/search", response_model=MemorySearchResponse)
def search_memories(
    q: str,
    limit: int = 20,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> MemorySearchResponse:
    svc = MemoryService(db, settings)
    mode, results = svc.search(user, q, limit=limit)
    return MemorySearchResponse(query=q, mode=mode, results=results)


@router.get("/{memory_id}", response_model=MemoryResponse)
def get_memory(
    memory_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> MemoryResponse:
    return MemoryService(db, settings).get_memory(user, memory_id)
