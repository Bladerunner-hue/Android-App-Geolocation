"""Fail-closed configuration."""

from __future__ import annotations

import pytest


def test_jwt_secret_required(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("GEO_TEST_MODE", "0")
    monkeypatch.setenv("GEO_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    from backend.app.config import Settings, clear_settings_cache

    clear_settings_cache()
    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    # Construct without env file
    s = Settings.model_construct(
        geo_database_url="sqlite+pysqlite:///:memory:",
        database_url="",
        jwt_secret="",
        test_mode=False,
    )
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        _ = s.resolved_jwt_secret


def test_placeholder_jwt_rejected():
    from backend.app.config import Settings

    s = Settings.model_construct(
        geo_database_url="sqlite+pysqlite:///:memory:",
        database_url="",
        jwt_secret="change-me",
        test_mode=False,
    )
    with pytest.raises(RuntimeError, match="placeholder"):
        _ = s.resolved_jwt_secret


def test_database_url_required():
    from backend.app.config import Settings

    s = Settings.model_construct(
        geo_database_url="",
        database_url="",
        jwt_secret="ok-secret-long-enough",
        test_mode=False,
    )
    with pytest.raises(RuntimeError, match="GEO_DATABASE_URL"):
        _ = s.resolved_database_url


def test_test_mode_allows_jwt():
    from backend.app.config import Settings

    s = Settings.model_construct(
        geo_database_url="sqlite+pysqlite:///:memory:",
        database_url="",
        jwt_secret="",
        test_mode=True,
    )
    assert "test-only" in s.resolved_jwt_secret
