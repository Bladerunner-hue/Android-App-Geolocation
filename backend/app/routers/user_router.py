"""User profile aggregates — SQL only, never Spark."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.config import Settings, get_settings
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import VibeProfileResponse
from backend.app.services.memory_service import MemoryService

router = APIRouter(prefix="/api/user", tags=["user"])


@router.get("/vibe-profile", response_model=VibeProfileResponse)
def vibe_profile(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> VibeProfileResponse:
    data = MemoryService(db, settings).vibe_profile(user)
    return VibeProfileResponse(**data)
