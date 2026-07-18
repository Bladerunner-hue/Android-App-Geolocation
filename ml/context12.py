"""
Context contract: context12-v1 (exactly 12 float32 features).

Freeze this before collecting training features — Python, backend, and Android
must match bit-for-bit on the same inputs.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Union

import numpy as np

CONTEXT_REVISION = "context12-v1"
CONTEXT_DIM = 12

# Feature index legend (for docs / Android parity tests)
FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "utc_offset_norm",
    "lat_norm",
    "lon_sin",
    "lon_cos",
    "accuracy_norm",
    "has_location",
]


def context12(
    captured_at_utc: datetime,
    utc_offset_minutes: int,
    latitude: Optional[float],
    longitude: Optional[float],
    accuracy_m: Optional[float],
) -> np.ndarray:
    """Return float32 vector of shape (12,)."""
    if captured_at_utc.tzinfo is None:
        captured_at_utc = captured_at_utc.replace(tzinfo=timezone.utc)
    local = captured_at_utc.astimezone(timezone(timedelta(minutes=int(utc_offset_minutes))))
    hour = local.hour + local.minute / 60.0 + local.second / 3600.0
    dow = float(local.weekday())
    doy = local.timetuple().tm_yday - 1 + hour / 24.0

    has_location = latitude is not None and longitude is not None
    lat = float(np.clip(latitude / 90.0, -1.0, 1.0)) if has_location else 0.0
    lon_rad = math.radians(float(longitude)) if has_location else 0.0
    accuracy = 0.0
    if has_location:
        accuracy = min(
            math.log1p(max(float(accuracy_m or 0.0), 0.0)) / math.log1p(5000.0),
            1.0,
        )

    return np.asarray(
        [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            math.sin(2 * math.pi * doy / 365.2425),
            math.cos(2 * math.pi * doy / 365.2425),
            float(np.clip(utc_offset_minutes / 840.0, -1.0, 1.0)),
            lat,
            math.sin(lon_rad) if has_location else 0.0,
            math.cos(lon_rad) if has_location else 0.0,
            accuracy,
            float(has_location),
        ],
        dtype=np.float32,
    )


def modality_mask(
    photo_is_present: bool,
    audio_is_present: bool,
    time_always: bool = True,
) -> np.ndarray:
    """Binary mask [photo, audio, time]. Time is always 1.0 for capture events."""
    return np.asarray(
        [
            float(photo_is_present),
            float(audio_is_present),
            1.0 if time_always else 0.0,
        ],
        dtype=np.float32,
    )


def zero_missing(
    image_emb: np.ndarray,
    audio_emb: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Missing modality → zero vector AND zero mask bit (caller sets mask)."""
    img = image_emb.astype(np.float32) * float(mask[0])
    aud = audio_emb.astype(np.float32) * float(mask[1])
    return img, aud


def context12_batch(
    rows: Sequence[dict],
) -> np.ndarray:
    """rows: dicts with keys captured_at_utc, utc_offset_minutes, latitude, longitude, accuracy_m."""
    return np.stack(
        [
            context12(
                r["captured_at_utc"],
                int(r["utc_offset_minutes"]),
                r.get("latitude"),
                r.get("longitude"),
                r.get("accuracy_m"),
            )
            for r in rows
        ],
        axis=0,
    )


if __name__ == "__main__":
    v = context12(datetime.now(timezone.utc), 120, 48.85, 2.35, 12.0)
    assert v.shape == (12,)
    assert v.dtype == np.float32
    m = modality_mask(True, False)
    assert m.tolist() == [1.0, 0.0, 1.0]
    print(CONTEXT_REVISION, v)
