"""GeoJournal FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from backend.app.auth import hash_password
from backend.app.config import get_settings
from backend.app.database import create_all, get_session_factory, init_db
from backend.app.models import User
from backend.app.routers import auth_router, memories, user_router
from backend.app.routers import training as training_router

logger = logging.getLogger(__name__)


def _seed_demo_user_if_enabled() -> None:
    """Create a local demo account when GEO_SEED_DEMO_USER=1 (dev only)."""
    settings = get_settings()
    if not settings.seed_demo_user:
        return
    password = (settings.demo_password or "").strip()
    if not password or password in {"CHANGE_ME", "demo-pass-change-me", "password"}:
        logger.warning(
            "GEO_SEED_DEMO_USER set but GEO_DEMO_PASSWORD is missing or a placeholder; skip seed."
        )
        return
    factory = get_session_factory()
    with factory() as db:
        existing = db.scalar(select(User).where(User.username == settings.demo_username))
        if existing is not None:
            return
        user = User(
            username=settings.demo_username,
            email=settings.demo_email,
            password_hash=hash_password(password),
        )
        db.add(user)
        db.commit()
        logger.info("Seeded demo user %r (local only).", settings.demo_username)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_db(settings)
    create_all()
    settings.media_root.mkdir(parents=True, exist_ok=True)
    _seed_demo_user_if_enabled()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="GeoJournal API",
        version="0.1.0",
        description="Offline-first memory journal backend. JWT-isolated, fail-closed.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router.router)
    app.include_router(memories.router)
    app.include_router(user_router.router)
    app.include_router(training_router.router)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "geojournal-api",
            "embedding_contract": {
                "perceptual": 128,
                "semantic_e5": 1024,
                "text": 1024,
                "insight": 128,
                "audio_yamnet": 1024,
                "image_mobilenet": 576,
                "semantic_model": "intfloat/e5-large-v2",
            },
        }

    return app


app = create_app()
