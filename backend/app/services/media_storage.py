"""Bounded, user-isolated media storage under GEO_MEDIA_ROOT.

Streams chunks while hashing; stops at max size; sniffs magic bytes.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException, UploadFile, status

from backend.app.config import Settings

# Magic-byte prefixes (not exhaustive; enough for journal media)
_JPEG = b"\xff\xd8\xff"
_PNG = b"\x89PNG\r\n\x1a\n"
_WEBP_RIFF = b"RIFF"
_WAV = b"RIFF"


def _sniff_kind(header: bytes, kind: str) -> bool:
    if kind == "photo":
        if header.startswith(_JPEG) or header.startswith(_PNG):
            return True
        if len(header) >= 12 and header.startswith(_WEBP_RIFF) and header[8:12] == b"WEBP":
            return True
        return False
    if kind == "audio":
        # WAV: RIFF....WAVE
        if len(header) >= 12 and header.startswith(_WAV) and header[8:12] == b"WAVE":
            return True
        # bare PCM from Android recorder may have no header — allow empty sniff for .wav path only
        return len(header) == 0
    return False


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
        """Returns (relative_path, sha256_hex) or (None, None). Streams; never load whole body first."""
        if file is None or not file.filename:
            return None, None

        content_type = (file.content_type or "").split(";")[0].strip().lower()
        ext = Path(file.filename).suffix.lower()
        if content_type and content_type not in allowed_content:
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

        hasher = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        header = b""
        while True:
            piece = await file.read(64 * 1024)
            if not piece:
                break
            if total == 0:
                header = piece[:16]
            total += len(piece)
            if total > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"{kind} exceeds max size {max_bytes} bytes",
                )
            hasher.update(piece)
            chunks.append(piece)

        if total == 0:
            return None, None

        if not _sniff_kind(header, kind):
            # Allow client .wav without WAVE header (raw PCM) for ambient capture
            if not (kind == "audio" and ext in {".wav", ".pcm"}):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"{kind} content does not match expected magic bytes",
                )

        digest = hasher.hexdigest()
        out_ext = ext or (".jpg" if kind == "photo" else ".wav")
        name = f"{kind}_{uuid.uuid4().hex}{out_ext}"
        dest = self._user_dir(user_id) / name
        with dest.open("wb") as fh:
            for c in chunks:
                fh.write(c)
        rel = str(dest.relative_to(self.root))
        return rel, digest

    def delete_relative(self, relative_path: Optional[str]) -> None:
        if not relative_path:
            return
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return
        if path.is_file():
            path.unlink(missing_ok=True)
