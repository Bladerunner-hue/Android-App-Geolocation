"""GeoJournal FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.database import create_all, init_db
from backend.app.routers import auth_router, memories, user_router


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_db(settings)
    create_all()
    settings.media_root.mkdir(parents=True, exist_ok=True)
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

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "geojournal-api"}

    return app


app = create_app()
