"""Train Mode label intake.

Backend never trains. Never invents vibes.
Cloud storage requires consent_for_cloud=true (you are already leaving the device).
consent_for_training is stored for Bronze export filters later.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import TrainingLabel, User
from backend.app.schemas import TrainingLabelCreate, TrainingLabelResponse


def _to_response(row: TrainingLabel) -> TrainingLabelResponse:
    secondary = row.secondary_vibes
    if not isinstance(secondary, list):
        secondary = []
    return TrainingLabelResponse(
        id=str(row.id),
        memory_id=row.memory_id,
        user_id=row.user_id,
        session_id=row.session_id,
        primary_vibe=row.primary_vibe,
        secondary_vibes=[str(x) for x in secondary],
        valence=row.valence,
        arousal=row.arousal,
        label_confidence=row.label_confidence,
        label_source=row.label_source,
        utc_offset_minutes=row.utc_offset_minutes,
        location_accuracy_m=row.location_accuracy_m,
        consent_for_training=row.consent_for_training,
        consent_for_cloud=row.consent_for_cloud,
        labelled_at=row.labelled_at,
        corrects_label_id=str(row.corrects_label_id) if row.corrects_label_id else None,
        created_at=row.created_at,
    )


class TrainingService:
    def __init__(self, db: Session):
        self.db = db

    def create_label(self, user: User, body: TrainingLabelCreate) -> TrainingLabelResponse:
        if not body.consent_for_cloud:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "consent_for_cloud must be true to store a label on the server. "
                    "Keep labels on-device when cloud consent is denied."
                ),
            )

        existing = self.db.get(TrainingLabel, str(body.id))
        if existing is not None:
            if existing.user_id != user.id:
                raise HTTPException(status_code=403, detail="Label id belongs to another user")
            return _to_response(existing)

        if body.corrects_label_id is not None:
            prior = self.db.get(TrainingLabel, str(body.corrects_label_id))
            if prior is None or prior.user_id != user.id:
                raise HTTPException(
                    status_code=400,
                    detail="corrects_label_id not found for this user",
                )

        row = TrainingLabel(
            id=str(body.id),
            memory_id=body.memory_id,
            user_id=user.id,
            session_id=body.session_id,
            primary_vibe=body.primary_vibe,
            secondary_vibes=list(body.secondary_vibes),
            valence=body.valence,
            arousal=body.arousal,
            label_confidence=body.label_confidence,
            label_source=body.label_source,
            utc_offset_minutes=body.utc_offset_minutes,
            location_accuracy_m=body.location_accuracy_m,
            consent_for_training=body.consent_for_training,
            consent_for_cloud=True,
            labelled_at=body.labelled_at or datetime.utcnow(),
            corrects_label_id=(
                str(body.corrects_label_id) if body.corrects_label_id else None
            ),
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return _to_response(row)

    def list_for_memory(
        self,
        user: User,
        memory_id: str,
        *,
        training_only: bool = False,
    ) -> List[TrainingLabelResponse]:
        q = select(TrainingLabel).where(
            TrainingLabel.user_id == user.id,
            TrainingLabel.memory_id == memory_id,
        )
        if training_only:
            q = q.where(TrainingLabel.consent_for_training.is_(True))
        q = q.order_by(TrainingLabel.labelled_at.desc())
        rows = list(self.db.scalars(q))
        return [_to_response(r) for r in rows]

    def get(self, user: User, label_id: UUID | str) -> TrainingLabelResponse:
        row = self.db.get(TrainingLabel, str(label_id))
        if row is None or row.user_id != user.id:
            raise HTTPException(status_code=404, detail="Label not found")
        return _to_response(row)
