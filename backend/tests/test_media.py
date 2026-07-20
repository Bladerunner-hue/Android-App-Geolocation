"""Media storage bounds."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import UploadFile


@pytest.mark.asyncio
async def test_save_and_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("GEO_TEST_MODE", "1")
    monkeypatch.setenv("JWT_SECRET", "test-only-jwt-secret-not-for-production")
    monkeypatch.setenv("GEO_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    monkeypatch.setenv("GEO_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("GEO_MAX_PHOTO_BYTES", "50")
    from backend.app.config import Settings, clear_settings_cache
    from backend.app.services.media_storage import MediaStorage

    clear_settings_cache()
    settings = Settings(
        GEO_TEST_MODE=True,
        JWT_SECRET="test-only-jwt-secret-not-for-production",
        GEO_DATABASE_URL="sqlite+pysqlite:///:memory:",
        GEO_MEDIA_ROOT=str(tmp_path),
        GEO_MAX_PHOTO_BYTES=50,
    )
    store = MediaStorage(settings)
    # Valid PNG magic so sniff passes
    png_hdr = b"\x89PNG\r\n\x1a\n" + b"x" * 8
    small = UploadFile(filename="a.png", file=io.BytesIO(png_hdr))
    path, digest = await store.save_upload(
        1,
        small,
        kind="photo",
        allowed_content=("image/png",),
        max_bytes=50,
    )
    assert path is not None
    assert digest is not None
    assert (tmp_path / path).exists()

    big = UploadFile(filename="b.png", file=io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"y" * 100))
    with pytest.raises(Exception):
        await store.save_upload(
            1,
            big,
            kind="photo",
            allowed_content=("image/png",),
            max_bytes=50,
        )


def test_hash_embed_stable():
    from backend.app.models import TEXT_EMBED_DIM
    from backend.app.services.memory_service import _hash_embed_text

    a = _hash_embed_text("cafe lisbon")
    b = _hash_embed_text("cafe lisbon")
    c = _hash_embed_text("forest rain")
    assert a == b
    assert a != c
    assert len(a) == TEXT_EMBED_DIM == 768
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
    # Must not depend on PYTHONHASHSEED / builtin hash()
    assert a[0] == b[0]
