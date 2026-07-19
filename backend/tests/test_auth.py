"""Auth tests."""

from __future__ import annotations


def test_register_and_login(client):
    r = client.post(
        "/api/auth/register",
        json={"username": "bob", "email": "bob@example.com", "password": "password123"},
    )
    assert r.status_code == 201
    assert r.json()["username"] == "bob"

    r = client.post(
        "/api/auth/login",
        json={"username": "bob", "password": "password123"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "token" in body
    assert body["user"]["email"] == "bob@example.com"


def test_register_duplicate_username(client):
    payload = {"username": "dup", "email": "a@example.com", "password": "password123"}
    assert client.post("/api/auth/register", json=payload).status_code == 201
    payload2 = {"username": "dup", "email": "b@example.com", "password": "password123"}
    assert client.post("/api/auth/register", json=payload2).status_code == 400


def test_login_bad_password(client):
    client.post(
        "/api/auth/register",
        json={"username": "c", "email": "c@example.com", "password": "password123"},
    )
    r = client.post("/api/auth/login", json={"username": "c", "password": "wrongpass1"})
    assert r.status_code == 401


def test_protected_route_requires_token(client):
    r = client.get("/api/user/vibe-profile")
    assert r.status_code in (401, 403)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_seed_demo_user(tmp_path, monkeypatch):
    """GEO_SEED_DEMO_USER creates a local account when password is non-placeholder."""
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    monkeypatch.setenv("GEO_TEST_MODE", "1")
    monkeypatch.setenv("JWT_SECRET", "test-only-jwt-secret-not-for-production")
    monkeypatch.setenv("GEO_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'seed.db'}")
    monkeypatch.setenv("GEO_MEDIA_ROOT", str(tmp_path / "media"))
    monkeypatch.setenv("GEO_SEED_DEMO_USER", "1")
    monkeypatch.setenv("GEO_DEMO_USERNAME", "demo")
    monkeypatch.setenv("GEO_DEMO_EMAIL", "demo@localhost")
    monkeypatch.setenv("GEO_DEMO_PASSWORD", "local-demo-secret-9x7k")

    from backend.app.config import clear_settings_cache, get_settings
    from backend.app.database import Base, create_all, get_engine, init_db
    from backend.app.main import create_app

    clear_settings_cache()
    init_db(get_settings())
    Base.metadata.drop_all(bind=get_engine())
    create_all()

    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/api/auth/login",
            json={"username": "demo", "password": "local-demo-secret-9x7k"},
        )
        assert r.status_code == 200, r.text
        assert "token" in r.json()

    clear_settings_cache()
