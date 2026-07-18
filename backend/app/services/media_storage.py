"""Bounded, user-isolated media storage under GEO_MEDIA_ROOT."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException, UploadFile, status

from backend.app.config import Settings


class MediaStorage:
    def __init__(self, settings: Settings):
        self.root = settings.media_root
        self.max_photo = settings.max_photo_bytes
        self.max_audio = settings.max_audio_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _user_dir(self, user_id: int) -> Path:
        d = self.root / f"user_{user_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def save_upload(
        self,
        user_id: int,
        file: Optional[UploadFile],
        *,
        kind: str,
        allowed_content: Tuple[str, ...],
        max_bytes: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Returns (relative_path, sha256_hex) or (None, None)."""
        if file is None or not file.filename:
            return None, None
        content_type = (file.content_type or "").split(";")[0].strip().lower()
        if content_type and content_type not in allowed_content:
            # allow empty content-type from some clients; validate by extension
            ext = Path(file.filename).suffix.lower()
            if kind == "photo" and ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported photo type: {content_type or ext}",
                )
            if kind == "audio" and ext not in {".wav", ".pcm"}:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported audio type: {content_type or ext}",
                )

        data = await file.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{kind} exceeds max size {max_bytes} bytes",
            )
        if not data:
            return None, None

        digest = hashlib.sha256(data).hexdigest()
        ext = Path(file.filename).suffix.lower() or (".jpg" if kind == "photo" else ".wav")
        name = f"{kind}_{uuid.uuid4().hex}{ext}"
        dest = self._user_dir(user_id) / name
        dest.write_bytes(data)
        rel = str(dest.relative_to(self.root))
        return rel, digest
