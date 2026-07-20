"""Memory create/search/profile.

Rules:
  - Reject private_mode uploads hard.
  - Never invent vibes; store only client on-device results (+ media).
  - Enrichment only when GEO_ENRICHMENT_ENABLED and client request_enrichment.
  - No Spark. No silent external LLM. No training in the request path.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.models import (
    ANALYSIS_SOURCES,
    TEXT_EMBED_DIM,
    VIBE_LABELS,
    Memory,
    User,
)
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


def _stable_token_bucket(tok: str, dim: int) -> int:
    """Process-stable bucket (unlike Python's randomized hash())."""
    digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _hash_embed_text(text: str, dim: int = TEXT_EMBED_DIM) -> List[float]:
    """Deterministic bag-of-words placeholder embedding (NOT production semantic).

    Uses blake2b so vectors remain compatible across process restarts.
    Do not label search results as ``semantic`` while this is in use.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        vec[_stable_token_bucket(tok, dim)] += 1.0
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
        analysis_source=m.analysis_source or "unavailable",
        model_version=m.model_version,
        latitude=m.latitude,
        longitude=m.longitude,
        captured_at=m.captured_at,
        created_at=m.created_at,
        private_mode=m.private_mode,
        enrichment_requested=bool(m.enrichment_requested),
        evidence=m.evidence_json if isinstance(m.evidence_json, dict) else None,
        structured_evidence=(
            m.structured_evidence if isinstance(m.structured_evidence, dict) else None
        ),
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
            allowed_content=(
                "audio/wav",
                "audio/x-wav",
                "audio/wave",
                "application/octet-stream",
            ),
            max_bytes=self.settings.max_audio_bytes,
        )

        vibe = meta.on_device_vibe if meta.on_device_vibe in VIBE_LABELS else None
        conf = meta.on_device_confidence if vibe is not None else None

        # Client-reported source; never invent inference server-side here.
        if meta.analysis_source and meta.analysis_source in ANALYSIS_SOURCES:
            analysis_source = meta.analysis_source
        elif vibe is not None:
            analysis_source = "on_device"
        else:
            analysis_source = "unavailable"

        status_s = "unavailable"
        if vibe is not None:
            status_s = "on_device"

        enrichment_requested = bool(meta.request_enrichment)
        if (
            enrichment_requested
            and self.settings.enrichment_enabled
            and status_s == "unavailable"
        ):
            # Gated placeholder only — no silent LLM, no invented vibe.
            status_s = "pending"

        structured = dict(meta.structured_evidence or {})
        if meta.on_device_probs is not None and "vibe_probs" not in structured:
            structured["vibe_probs"] = meta.on_device_probs
        structured.setdefault("has_photo", photo_path is not None)
        structured.setdefault("has_audio", audio_path is not None)
        structured.setdefault(
            "has_location",
            meta.latitude is not None and meta.longitude is not None,
        )
        structured.setdefault("source", analysis_source)

        # Legacy free-form blob for older clients / tools
        evidence: Dict[str, Any] = {
            "has_photo": photo_path is not None,
            "has_audio": audio_path is not None,
            "has_location": meta.latitude is not None and meta.longitude is not None,
            "source": analysis_source,
            "vibe_probs": meta.on_device_probs,
        }

        content_sha = photo_sha or audio_sha
        text_blob = " ".join(
            filter(None, [meta.caption, vibe, f"{meta.latitude},{meta.longitude}"])
        )
        # Placeholder hash (1024-D to match E5 width). Prefer memory_semantic_embeddings
        # filled by GEO_E5_BASE_URL backfill — do not treat as real semantic quality.
        text_emb = _hash_embed_text(text_blob) if text_blob.strip() else None

        mem = Memory(
            user_id=user.id,
            client_uuid=meta.client_uuid,
            caption=meta.caption,
            vibe_label=vibe,
            vibe_confidence=conf,
            analysis_status=status_s,
            analysis_source=analysis_source,
            model_version=meta.model_version,
            latitude=meta.latitude,
            longitude=meta.longitude,
            captured_at=meta.captured_at or datetime.utcnow(),
            private_mode=False,
            photo_path=photo_path,
            audio_path=audio_path,
            perceptual_embedding=meta.perceptual_embedding,
            insight_embedding=meta.insight_embedding,
            text_embedding=text_emb,
            evidence_json=evidence,
            structured_evidence=structured or None,
            enrichment_requested=enrichment_requested,
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

    def delete_memory(self, user: User, memory_id: int) -> None:
        """Hard-delete owned memory and media files (privacy-first journal)."""
        mem = self.db.get(Memory, memory_id)
        if mem is None or mem.user_id != user.id:
            raise HTTPException(status_code=404, detail="Memory not found")
        self.media.delete_relative(mem.photo_path)
        self.media.delete_relative(mem.audio_path)
        self.db.delete(mem)
        self.db.commit()

    def search(self, user: User, q: str, limit: int = 20) -> Tuple[str, List[MemoryResponse]]:
        """Primary path: SQL lexical search. Hash-vector ranking is labeled
        ``lexical_placeholder`` — never ``semantic`` until E5 vectors are in
        memory_semantic_embeddings (intfloat/e5-large-v2, 1024-D).
        """
        q = (q or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Query q is required")
        limit = max(1, min(limit, 50))

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
        if lex:
            return "lexical", [memory_to_response(m) for m in lex]

        # Fallback ranking only when lexical misses and stored placeholder vectors exist.
        rows = list(
            self.db.scalars(
                select(Memory)
                .where(Memory.user_id == user.id)
                .order_by(Memory.captured_at.desc())
            )
        )
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
                return "lexical_placeholder", top

        return "lexical", []

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
