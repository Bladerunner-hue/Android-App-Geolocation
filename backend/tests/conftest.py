"""Pytest fixtures: SQLite + test JWT, isolated media root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Repo root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["GEO_TEST_MODE"] = "1"
os.environ["JWT_SECRET"] = "test-only-jwt-secret-not-for-production"
os.environ["GEO_DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["GEO_ENRICHMENT_ENABLED"] = "0"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GEO_TEST_MODE", "1")
    monkeypatch.setenv("JWT_SECRET", "test-only-jwt-secret-not-for-production")
    monkeypatch.setenv("GEO_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("GEO_MEDIA_ROOT", str(tmp_path / "media"))
    monkeypatch.setenv("GEO_ENRICHMENT_ENABLED", "0")

    from backend.app.config import clear_settings_cache, get_settings
    from backend.app.database import Base, create_all, get_engine, init_db

    clear_settings_cache()
    settings = get_settings()
    init_db(settings)
    Base.metadata.drop_all(bind=get_engine())
    create_all()

    from backend.app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    clear_settings_cache()


@pytest.fixture()
def auth_headers(client: TestClient) -> dict:
    r = client.post(
        "/api/auth/register",
        json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123",
        },
    )
    assert r.status_code == 201, r.text
    r = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "password123"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}
