"""Fail-closed settings: no default database password or JWT secret."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_REPO_ROOT / ".env"), str(_BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Prefer GEO_* names; DATABASE_URL accepted for local convenience.
    geo_database_url: str = Field(alias="GEO_DATABASE_URL", default="")
    database_url: str = Field(alias="DATABASE_URL", default="")
    jwt_secret: str = Field(alias="JWT_SECRET", default="")
    jwt_algorithm: str = Field(alias="JWT_ALGORITHM", default="HS256")
    access_token_expire_minutes: int = Field(
        alias="ACCESS_TOKEN_EXPIRE_MINUTES", default=60 * 24 * 7
    )
    media_root: Path = Field(
        alias="GEO_MEDIA_ROOT",
        default=_BACKEND_DIR / "media_store",
    )
    max_photo_bytes: int = Field(alias="GEO_MAX_PHOTO_BYTES", default=8 * 1024 * 1024)
    max_audio_bytes: int = Field(alias="GEO_MAX_AUDIO_BYTES", default=1 * 1024 * 1024)
    cors_origins: str = Field(alias="GEO_CORS_ORIGINS", default="*")
    enrichment_enabled: bool = Field(alias="GEO_ENRICHMENT_ENABLED", default=False)
    bind_host: str = Field(alias="GEO_BIND_HOST", default="127.0.0.1")
    # Testing: allow sqlite without secrets when GEO_TEST_MODE=1
    test_mode: bool = Field(alias="GEO_TEST_MODE", default=False)

    @property
    def resolved_database_url(self) -> str:
        url = (self.geo_database_url or self.database_url or "").strip()
        if not url:
            raise RuntimeError(
                "GEO_DATABASE_URL (or DATABASE_URL) is required. "
                "Copy backend/.env.example and set a real connection string."
            )
        return url

    @property
    def resolved_jwt_secret(self) -> str:
        secret = (self.jwt_secret or "").strip()
        if not secret:
            if self.test_mode:
                return "test-only-jwt-secret-not-for-production"
            raise RuntimeError(
                "JWT_SECRET is required. Copy backend/.env.example and set a strong secret."
            )
        if not self.test_mode and secret in {
            "change-me",
            "geotrack-super-secret-key-change-in-production",
            "secret",
        }:
            raise RuntimeError("JWT_SECRET must not use a known default placeholder.")
        return secret

    @property
    def cors_origin_list(self) -> List[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [p.strip() for p in raw.split(",") if p.strip()]

    @field_validator("media_root", mode="before")
    @classmethod
    def _path(cls, v: object) -> Path:
        return Path(str(v)).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
