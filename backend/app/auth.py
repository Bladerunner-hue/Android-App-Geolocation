"""JWT auth helpers — ownership always derived from token subject."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash

from backend.app.config import Settings, get_settings
from backend.app.database import get_db
from backend.app.models import User

security = HTTPBearer(auto_error=True)


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def create_access_token(
    subject: str,
    settings: Optional[Settings] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    settings = settings or get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload: Dict[str, Any] = {"sub": subject, "exp": expire}
    if extra:
        payload.update(extra)
    return jwt.encode(
        payload, settings.resolved_jwt_secret, algorithm=settings.jwt_algorithm
    )


def decode_token(token: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    settings = settings or get_settings()
    try:
        return jwt.decode(
            token,
            settings.resolved_jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    payload = decode_token(credentials.credentials, settings)
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid token subject")
    try:
        user_id = int(sub)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token subject") from exc
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive or missing")
    return user
