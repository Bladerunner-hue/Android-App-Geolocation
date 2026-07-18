"""SQLAlchemy models. Embeddings stored as JSON for SQLite parity; pgvector SQL migration for prod."""

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
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base

VIBE_LABELS = (
    "serene",
    "energetic",
    "chaotic",
    "nostalgic",
    "tense",
    "social",
    "contemplative",
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
    analysis_status: Mapped[str] = mapped_column(String(32), default="unavailable")
    # unavailable | on_device | enriched | pending
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    private_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    photo_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    audio_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # Separate vectors: perceptual (image/audio fusion) vs text (search encoder)
    perceptual_embedding: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    text_embedding: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    evidence_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    content_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    user: Mapped["User"] = relationship(back_populates="memories")
