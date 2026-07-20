"""SQLAlchemy models.

Postgres production types live in migrations (pgvector). ORM columns use JSON for
embeddings so GEO_TEST_MODE SQLite stays dependency-light. Application code owns
vector dimensions:

  perceptual_embedding  128   — fusion_v0 (NOT E5)
  text_embedding       1024   — E5 e5-large-v2 placeholder column / legacy
  semantic table       1024   — memory_semantic_embeddings (canonical E5 store)
  insight_embedding     128   — optional auxiliary
  YAMNet audio         1024   — different space from E5 text
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base

try:
    from pgvector.sqlalchemy import Vector as PgVector
except ImportError:  # pragma: no cover
    PgVector = None  # type: ignore[misc, assignment]


class EmbeddingVector(TypeDecorator):
    """JSON list on SQLite (tests); pgvector on PostgreSQL (production)."""

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and PgVector is not None:
            return dialect.type_descriptor(PgVector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return list(value)

# Must match Kotlin Train Mode + fusion_v0 head. Never invent labels outside this set.
VIBE_LABELS = (
    "serene",
    "energetic",
    "chaotic",
    "nostalgic",
    "tense",
    "social",
    "contemplative",
)

PERCEPTUAL_DIM = 128
# E5 intfloat/e5-large-v2 via HTTP :6100 — matches memory_semantic_embeddings
TEXT_EMBED_DIM = 1024
SEMANTIC_DIM = 1024
INSIGHT_DIM = 128
AUDIO_YAMNET_DIM = 1024
IMAGE_MOBILENET_DIM = 576
SEMANTIC_MODEL_ID = "intfloat/e5-large-v2"

ANALYSIS_SOURCES = frozenset(
    {"unavailable", "on_device", "server_fusion", "rules"}
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    memories: Mapped[list["Memory"]] = relationship(back_populates="user")
    training_labels: Mapped[list["TrainingLabel"]] = relationship(back_populates="user")


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        UniqueConstraint("user_id", "client_uuid", name="uq_memory_user_client_uuid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    client_uuid: Mapped[str] = mapped_column(String(64), index=True)
    caption: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    vibe_label: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    vibe_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # unavailable | on_device | enriched | pending — high-level status for clients
    analysis_status: Mapped[str] = mapped_column(String(32), default="unavailable")
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    private_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    photo_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    audio_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Separate embedding spaces (JSON on SQLite; pgvector on Postgres)
    perceptual_embedding: Mapped[Optional[Any]] = mapped_column(
        EmbeddingVector(PERCEPTUAL_DIM), nullable=True
    )
    text_embedding: Mapped[Optional[Any]] = mapped_column(
        EmbeddingVector(TEXT_EMBED_DIM), nullable=True
    )
    insight_embedding: Mapped[Optional[Any]] = mapped_column(
        EmbeddingVector(INSIGHT_DIM), nullable=True
    )

    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # on_device | server_fusion | rules | unavailable
    analysis_source: Mapped[str] = mapped_column(String(32), default="unavailable")
    # Legacy free-form evidence (kept for older clients)
    evidence_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    # Production structured evidence (vibe_probs, modality_mask, context12, …)
    structured_evidence: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    enrichment_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    content_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    user: Mapped["User"] = relationship(back_populates="memories")


class TrainingLabel(Base):
    """Human Train Mode labels. Backend never invents vibes; never trains here."""

    __tablename__ = "training_labels"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Kotlin memory client_uuid or server-side stable id string
    memory_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    primary_vibe: Mapped[str] = mapped_column(String(32))
    secondary_vibes: Mapped[Any] = mapped_column(JSON, default=list)
    valence: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    arousal: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    label_confidence: Mapped[int] = mapped_column(SmallInteger)
    label_source: Mapped[str] = mapped_column(String(32), default="human_self")
    utc_offset_minutes: Mapped[int] = mapped_column(Integer)
    location_accuracy_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    consent_for_training: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_for_cloud: Mapped[bool] = mapped_column(Boolean, default=False)
    labelled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    corrects_label_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="training_labels")
