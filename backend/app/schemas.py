"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str


class LoginResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    user: UserResponse


class MemoryEvidence(BaseModel):
    has_photo: bool = True
    has_audio: bool = False
    has_location: bool = False
    vibe_probs: Optional[List[float]] = None
    source: str = "unavailable"  # unavailable | on_device | server


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    client_uuid: str
    caption: Optional[str] = None
    vibe_label: Optional[str] = None
    vibe_confidence: Optional[float] = None
    analysis_status: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    captured_at: datetime
    created_at: datetime
    private_mode: bool
    evidence: Optional[Dict[str, Any]] = None


class MemorySearchResponse(BaseModel):
    query: str
    mode: str  # semantic | lexical
    results: List[MemoryResponse]


class VibeProfileResponse(BaseModel):
    total_memories: int
    vibe_counts: Dict[str, int]
    top_vibes: List[str]


class AnalyzeMeta(BaseModel):
    client_uuid: str
    private_mode: bool = False
    caption: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    captured_at: Optional[datetime] = None
    on_device_vibe: Optional[str] = None
    on_device_confidence: Optional[float] = None
    on_device_probs: Optional[List[float]] = None
    request_enrichment: bool = False
