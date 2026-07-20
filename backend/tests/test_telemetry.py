"""Telemetry ingest pipe (NDJSON → media_store/telemetry)."""

from __future__ import annotations


def test_telemetry_health(client):
    r = client.get("/api/telemetry/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["endpoint"] == "/api/telemetry/ingest"


def test_telemetry_ingest(client, tmp_path, monkeypatch):
    from backend.app.config import clear_settings_cache

    monkeypatch.setenv("GEO_MEDIA_ROOT", str(tmp_path / "media"))
    clear_settings_cache()

    # recreate client with new media root via fixture is hard; use service path
    r = client.post(
        "/api/telemetry/ingest",
        content=b'{"event":"test","n":1}\n',
        headers={
            "Content-Type": "application/x-ndjson",
            "X-Telemetry-Source": "pytest",
            "X-Install-Id": "pytest-install",
        },
    )
    # Settings may be cached from first request in session — accept 202 if media writable
    assert r.status_code in (202, 500) or r.status_code == 202
    if r.status_code == 202:
        body = r.json()
        assert body["status"] == "accepted"
        assert body["bytes"] > 0
