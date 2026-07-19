"""Train Mode label API. Storage only — no training, no inference."""

from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import TrainingLabelCreate, TrainingLabelResponse
from backend.app.services.training_service import TrainingService

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/labels", response_model=TrainingLabelResponse, status_code=201)
def create_training_label(
    body: TrainingLabelCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TrainingLabelResponse:
    return TrainingService(db).create_label(user, body)


@router.get("/labels/{label_id}", response_model=TrainingLabelResponse)
def get_training_label(
    label_id: UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TrainingLabelResponse:
    return TrainingService(db).get(user, label_id)


@router.get("/labels", response_model=List[TrainingLabelResponse])
def list_training_labels_for_memory(
    memory_id: str = Query(..., min_length=1, max_length=64),
    training_only: bool = Query(
        False,
        description="If true, only rows with consent_for_training=true",
    ),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[TrainingLabelResponse]:
    return TrainingService(db).list_for_memory(
        user, memory_id, training_only=training_only
    )
