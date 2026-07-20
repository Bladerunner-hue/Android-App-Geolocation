"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from backend.app.models import (
    ANALYSIS_SOURCES,
    INSIGHT_DIM,
    PERCEPTUAL_DIM,
    SEMANTIC_DIM,
    TEXT_EMBED_DIM,
    VIBE_LABELS,
)


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
    source: str = "unavailable"  # unavailable | on_device | server_fusion | rules


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    client_uuid: str
    caption: Optional[str] = None
    vibe_label: Optional[str] = None
    vibe_confidence: Optional[float] = None
    analysis_status: str
    analysis_source: str = "unavailable"
    model_version: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    captured_at: datetime
    created_at: datetime
    private_mode: bool
    enrichment_requested: bool = False
    # Prefer structured_evidence; evidence is legacy free-form
    evidence: Optional[Dict[str, Any]] = None
    structured_evidence: Optional[Dict[str, Any]] = None
    # Embeddings are not returned by default (size + privacy); omit from wire unless needed


class MemorySearchResponse(BaseModel):
    query: str
    mode: str  # semantic | lexical
    results: List[MemoryResponse]


class VibeProfileResponse(BaseModel):
    total_memories: int
    vibe_counts: Dict[str, int]
    top_vibes: List[str]


class AnalyzeMeta(BaseModel):
    """Client → server payload for /api/memories/analyze.

    Server stores on-device results; never invents a vibe.
    """

    client_uuid: str
    private_mode: bool = False
    caption: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    captured_at: Optional[datetime] = None
    on_device_vibe: Optional[str] = None
    on_device_confidence: Optional[float] = None
    on_device_probs: Optional[List[float]] = None
    # fusion_v0 perceptual 128-D (optional until TFLite is packaged)
    perceptual_embedding: Optional[List[float]] = None
    insight_embedding: Optional[List[float]] = None
    model_version: Optional[str] = None
    # on_device | server_fusion | rules | unavailable — client-reported source of vibe
    analysis_source: Optional[str] = None
    structured_evidence: Optional[Dict[str, Any]] = None
    request_enrichment: bool = False

    @field_validator("on_device_vibe")
    @classmethod
    def _vibe_known_or_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if v not in VIBE_LABELS:
            # Do not invent; drop unknown labels rather than 422 the whole upload
            return None
        return v

    @field_validator("perceptual_embedding")
    @classmethod
    def _perc_dim(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return None
        if len(v) != PERCEPTUAL_DIM:
            raise ValueError(f"perceptual_embedding must be length {PERCEPTUAL_DIM}")
        return v

    @field_validator("insight_embedding")
    @classmethod
    def _insight_dim(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return None
        if len(v) != INSIGHT_DIM:
            raise ValueError(f"insight_embedding must be length {INSIGHT_DIM}")
        return v

    @field_validator("on_device_probs")
    @classmethod
    def _probs_len(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return None
        if len(v) != len(VIBE_LABELS):
            raise ValueError(f"on_device_probs must be length {len(VIBE_LABELS)}")
        return v

    @field_validator("analysis_source")
    @classmethod
    def _source_ok(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if v not in ANALYSIS_SOURCES:
            raise ValueError(f"analysis_source must be one of {sorted(ANALYSIS_SOURCES)}")
        return v


class TrainingLabelCreate(BaseModel):
    """POST /api/training/labels — consented human labels only."""

    id: UUID
    memory_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    primary_vibe: str
    secondary_vibes: List[str] = Field(default_factory=list)
    valence: Optional[int] = Field(default=None, ge=-2, le=2)
    arousal: Optional[int] = Field(default=None, ge=1, le=5)
    label_confidence: int = Field(ge=1, le=3)
    label_source: str = Field(default="human_self", max_length=32)
    utc_offset_minutes: int
    location_accuracy_m: Optional[float] = None
    consent_for_training: bool = False
    consent_for_cloud: bool = False
    labelled_at: Optional[datetime] = None
    corrects_label_id: Optional[UUID] = None

    @field_validator("primary_vibe")
    @classmethod
    def _primary_vibe(cls, v: str) -> str:
        if v not in VIBE_LABELS:
            raise ValueError(f"primary_vibe must be one of {VIBE_LABELS}")
        return v

    @field_validator("secondary_vibes")
    @classmethod
    def _secondary(cls, v: List[str]) -> List[str]:
        for x in v:
            if x not in VIBE_LABELS:
                raise ValueError(f"secondary vibe {x!r} not in VIBE_LABELS")
        return v


class TrainingLabelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    memory_id: str
    user_id: int
    session_id: str
    primary_vibe: str
    secondary_vibes: List[str]
    valence: Optional[int] = None
    arousal: Optional[int] = None
    label_confidence: int
    label_source: str
    utc_offset_minutes: int
    location_accuracy_m: Optional[float] = None
    consent_for_training: bool
    consent_for_cloud: bool
    labelled_at: datetime
    corrects_label_id: Optional[str] = None
    created_at: datetime


# Document dim constants for OpenAPI consumers
class EmbeddingContract(BaseModel):
    perceptual_dim: int = PERCEPTUAL_DIM
    text_dim: int = TEXT_EMBED_DIM
    semantic_dim: int = SEMANTIC_DIM
    insight_dim: int = INSIGHT_DIM
    semantic_model: str = "intfloat/e5-large-v2"
    vibe_labels: tuple[str, ...] = VIBE_LABELS
