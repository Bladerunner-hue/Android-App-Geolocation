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
