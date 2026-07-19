"""Train Mode label API — consent-gated, no invented vibes."""

from __future__ import annotations

import uuid


def test_training_label_requires_cloud_consent(client, auth_headers):
    body = {
        "id": str(uuid.uuid4()),
        "memory_id": "mem-client-1",
        "session_id": "sess-1",
        "primary_vibe": "serene",
        "label_confidence": 2,
        "utc_offset_minutes": 120,
        "consent_for_training": True,
        "consent_for_cloud": False,
    }
    r = client.post("/api/training/labels", json=body, headers=auth_headers)
    assert r.status_code == 403, r.text


def test_training_label_create_and_idempotent(client, auth_headers):
    lid = str(uuid.uuid4())
    body = {
        "id": lid,
        "memory_id": "mem-client-2",
        "session_id": "sess-2",
        "primary_vibe": "contemplative",
        "secondary_vibes": ["serene"],
        "valence": 1,
        "arousal": 2,
        "label_confidence": 3,
        "label_source": "human_self",
        "utc_offset_minutes": 60,
        "location_accuracy_m": 12.5,
        "consent_for_training": True,
        "consent_for_cloud": True,
    }
    r1 = client.post("/api/training/labels", json=body, headers=auth_headers)
    assert r1.status_code == 201, r1.text
    data = r1.json()
    assert data["id"] == lid
    assert data["primary_vibe"] == "contemplative"
    assert data["consent_for_training"] is True
    assert data["secondary_vibes"] == ["serene"]

    r2 = client.post("/api/training/labels", json=body, headers=auth_headers)
    assert r2.status_code == 201
    assert r2.json()["id"] == lid

    r3 = client.get(f"/api/training/labels/{lid}", headers=auth_headers)
    assert r3.status_code == 200
    assert r3.json()["memory_id"] == "mem-client-2"

    listed = client.get(
        "/api/training/labels",
        params={"memory_id": "mem-client-2", "training_only": True},
        headers=auth_headers,
    )
    assert listed.status_code == 200
    assert any(x["id"] == lid for x in listed.json())


def test_training_label_rejects_unknown_vibe(client, auth_headers):
    body = {
        "id": str(uuid.uuid4()),
        "memory_id": "m",
        "session_id": "s",
        "primary_vibe": "happy",  # not in VIBE_LABELS
        "label_confidence": 1,
        "utc_offset_minutes": 0,
        "consent_for_cloud": True,
    }
    r = client.post("/api/training/labels", json=body, headers=auth_headers)
    assert r.status_code == 422


def test_training_label_isolation(client, auth_headers):
    lid = str(uuid.uuid4())
    body = {
        "id": lid,
        "memory_id": "iso-mem",
        "session_id": "iso-sess",
        "primary_vibe": "tense",
        "label_confidence": 1,
        "utc_offset_minutes": 0,
        "consent_for_cloud": True,
        "consent_for_training": False,
    }
    assert client.post("/api/training/labels", json=body, headers=auth_headers).status_code == 201

    client.post(
        "/api/auth/register",
        json={"username": "label_eve", "email": "label_eve@example.com", "password": "password123"},
    )
    tok = client.post(
        "/api/auth/login",
        json={"username": "label_eve", "password": "password123"},
    ).json()["token"]
    eve = {"Authorization": f"Bearer {tok}"}
    assert client.get(f"/api/training/labels/{lid}", headers=eve).status_code == 404
