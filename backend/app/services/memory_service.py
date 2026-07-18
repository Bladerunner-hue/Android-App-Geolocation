"""Memory create/search/profile. Never launches Spark. No silent cloud LLM."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.models import VIBE_LABELS, Memory, User
from backend.app.schemas import AnalyzeMeta, MemoryResponse
from backend.app.services.media_storage import MediaStorage


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return -1.0
    return dot / (na * nb)


def _hash_embed_text(text: str, dim: int = 64) -> List[float]:
    """Deterministic bag-of-words hash embedding for lexical-semantic bridge.
    Not a production text encoder — used until an aligned model is registered.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        h = hash(tok) % dim
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def memory_to_response(m: Memory) -> MemoryResponse:
    return MemoryResponse(
        id=m.id,
        client_uuid=m.client_uuid,
        caption=m.caption,
        vibe_label=m.vibe_label,
        vibe_confidence=m.vibe_confidence,
        analysis_status=m.analysis_status,
        latitude=m.latitude,
        longitude=m.longitude,
        captured_at=m.captured_at,
        created_at=m.created_at,
        private_mode=m.private_mode,
        evidence=m.evidence_json,
    )


class MemoryService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.media = MediaStorage(settings)

    async def analyze_upload(
        self,
        user: User,
        meta: AnalyzeMeta,
        photo: Optional[UploadFile],
        audio: Optional[UploadFile],
    ) -> Tuple[MemoryResponse, bool]:
        """Returns (memory, created_new). Rejects private_mode payloads defensively."""
        if meta.private_mode:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Private Mode memories must not be uploaded. Rejected by server.",
            )

        existing = self.db.scalar(
            select(Memory).where(
                Memory.user_id == user.id,
                Memory.client_uuid == meta.client_uuid,
            )
        )
        if existing is not None:
            return memory_to_response(existing), False

        photo_path, photo_sha = await self.media.save_upload(
            user.id,
            photo,
            kind="photo",
            allowed_content=("image/jpeg", "image/png", "image/webp", "image/jpg"),
            max_bytes=self.settings.max_photo_bytes,
        )
        audio_path, audio_sha = await self.media.save_upload(
            user.id,
            audio,
            kind="audio",
            allowed_content=("audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"),
            max_bytes=self.settings.max_audio_bytes,
        )

        vibe = meta.on_device_vibe
        conf = meta.on_device_confidence
        status_s = "unavailable"
        if vibe and vibe in VIBE_LABELS:
            status_s = "on_device"
        if (
            meta.request_enrichment
            and self.settings.enrichment_enabled
            and status_s == "unavailable"
        ):
            # Placeholder: no silent LLM. Optional local enrichment only when enabled.
            status_s = "pending"

        evidence: Dict[str, Any] = {
            "has_photo": photo_path is not None,
            "has_audio": audio_path is not None,
            "has_location": meta.latitude is not None and meta.longitude is not None,
            "source": status_s,
            "vibe_probs": meta.on_device_probs,
        }
        content_sha = photo_sha or audio_sha
        text_blob = " ".join(
            filter(None, [meta.caption, vibe, f"{meta.latitude},{meta.longitude}"])
        )
        text_emb = _hash_embed_text(text_blob) if text_blob.strip() else None

        mem = Memory(
            user_id=user.id,
            client_uuid=meta.client_uuid,
            caption=meta.caption,
            vibe_label=vibe if vibe in VIBE_LABELS else None,
            vibe_confidence=conf,
            analysis_status=status_s,
            latitude=meta.latitude,
            longitude=meta.longitude,
            captured_at=meta.captured_at or datetime.utcnow(),
            private_mode=False,
            photo_path=photo_path,
            audio_path=audio_path,
            text_embedding=text_emb,
            evidence_json=evidence,
            content_sha256=content_sha,
        )
        self.db.add(mem)
        self.db.commit()
        self.db.refresh(mem)
        return memory_to_response(mem), True

    def get_memory(self, user: User, memory_id: int) -> MemoryResponse:
        mem = self.db.get(Memory, memory_id)
        if mem is None or mem.user_id != user.id:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory_to_response(mem)

    def search(self, user: User, q: str, limit: int = 20) -> Tuple[str, List[MemoryResponse]]:
        q = (q or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Query q is required")
        limit = max(1, min(limit, 50))
        rows = list(
            self.db.scalars(
                select(Memory).where(Memory.user_id == user.id).order_by(Memory.captured_at.desc())
            )
        )
        # Prefer semantic when text embeddings exist
        with_emb = [m for m in rows if m.text_embedding]
        if with_emb:
            qv = _hash_embed_text(q)
            scored = sorted(
                ((_cosine(qv, m.text_embedding or []), m) for m in with_emb),
                key=lambda t: t[0],
                reverse=True,
            )
            top = [memory_to_response(m) for s, m in scored[:limit] if s > 0]
            if top:
                return "semantic", top

        # Lexical fallback
        like = f"%{q.lower()}%"
        lex = list(
            self.db.scalars(
                select(Memory)
                .where(
                    Memory.user_id == user.id,
                    or_(
                        func.lower(Memory.caption).like(like),
                        func.lower(Memory.vibe_label).like(like),
                    ),
                )
                .order_by(Memory.captured_at.desc())
                .limit(limit)
            )
        )
        return "lexical", [memory_to_response(m) for m in lex]

    def vibe_profile(self, user: User) -> Dict[str, Any]:
        """Fast SQL aggregate only — never launches Spark."""
        rows = list(
            self.db.execute(
                select(Memory.vibe_label, func.count())
                .where(Memory.user_id == user.id, Memory.vibe_label.is_not(None))
                .group_by(Memory.vibe_label)
            )
        )
        counts = {str(label): int(n) for label, n in rows if label}
        total = self.db.scalar(
            select(func.count()).select_from(Memory).where(Memory.user_id == user.id)
        ) or 0
        top = sorted(counts, key=counts.get, reverse=True)  # type: ignore[arg-type]
        return {
            "total_memories": int(total),
            "vibe_counts": counts,
            "top_vibes": top[:5],
        }
