"""Memory upload / search / get. No inference invention; no Spark."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.config import Settings, get_settings
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import AnalyzeMeta, MemoryResponse, MemorySearchResponse
from backend.app.services.memory_service import MemoryService

router = APIRouter(prefix="/api/memories", tags=["memories"])


def _json_list(raw: Optional[str], field: str) -> Optional[List[float]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field} must be valid JSON",
        ) from exc
    if not isinstance(data, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field} must be a JSON array",
        )
    return [float(x) for x in data]


def _json_obj(raw: Optional[str], field: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field} must be valid JSON",
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field} must be a JSON object",
        )
    return data


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
    perceptual_embedding: Optional[str],
    insight_embedding: Optional[str],
    model_version: Optional[str],
    analysis_source: Optional[str],
    structured_evidence: Optional[str],
    request_enrichment: bool,
) -> AnalyzeMeta:
    ts = None
    if captured_at:
        ts = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).replace(tzinfo=None)
    try:
        return AnalyzeMeta(
            client_uuid=client_uuid,
            private_mode=private_mode,
            caption=caption,
            latitude=latitude,
            longitude=longitude,
            captured_at=ts,
            on_device_vibe=on_device_vibe,
            on_device_confidence=on_device_confidence,
            on_device_probs=_json_list(on_device_probs, "on_device_probs"),
            perceptual_embedding=_json_list(perceptual_embedding, "perceptual_embedding"),
            insight_embedding=_json_list(insight_embedding, "insight_embedding"),
            model_version=model_version,
            analysis_source=analysis_source,
            structured_evidence=_json_obj(structured_evidence, "structured_evidence"),
            request_enrichment=request_enrichment,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


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
    perceptual_embedding: Optional[str] = Form(None),
    insight_embedding: Optional[str] = Form(None),
    model_version: Optional[str] = Form(None),
    analysis_source: Optional[str] = Form(None),
    structured_evidence: Optional[str] = Form(None),
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
        perceptual_embedding,
        insight_embedding,
        model_version,
        analysis_source,
        structured_evidence,
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


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(
    memory_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> None:
    """Privacy-first hard delete (row + media files)."""
    MemoryService(db, settings).delete_memory(user, memory_id)
