"""
Minimal-data bootstrap: synthetic multimodal TFRecords so training works day-1.

Real path later:
  PySpark Bronze → Silver → Gold → executor TFRecordWriter + manifest
  (see backend/jobs/pyspark_export_gold.py)

This script does NOT replace Spark at scale; it proves the TF pipeline with
~200 labelled-like examples and heavy on-the-fly augmentation.

Public bootstrap datasets you can swap in:
  - ESC-50 → map environmental sounds to vibe classes
  - Your app memories (photo + 5–10s ambient audio + lat/lon + user tag)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# Repo root on path (ml/experiments/ → parents[2])
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.config import (  # noqa: E402
    AUDIO_FRAMES,
    DEFAULT_TFRECORD_DIR,
    GEO_RAW_DIM,
    IMG_SIZE,
    N_MELS,
    NUM_COT_CLASSES,
    NUM_COT_SLOTS,
    NUM_VIBE_CLASSES,
    SCHEMA_VERSION,
    VIBE_LABELS,
    Manifest,
    feature_spec,
)

FEATURE_SPEC = feature_spec()


def _bytes_feature(v: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _float_feature(v: List[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))


def _int64_feature(v: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def _int64_list_feature(v: List[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _synth_image(vibe: int, rng: np.random.Generator) -> np.ndarray:
    """Class-correlated colour blobs so the model can actually learn."""
    img = rng.normal(0.45, 0.08, size=(IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    # Tint per vibe
    tints = np.array(
        [
            [0.2, 0.5, 0.3],  # serene green
            [0.7, 0.4, 0.1],  # energetic orange
            [0.5, 0.1, 0.5],  # chaotic purple
            [0.5, 0.4, 0.3],  # nostalgic sepia
            [0.5, 0.1, 0.1],  # tense red
            [0.3, 0.4, 0.7],  # social blue
            [0.25, 0.25, 0.4],  # contemplative indigo
        ],
        dtype=np.float32,
    )
    img = 0.55 * img + 0.45 * tints[vibe]
    # Add a soft circle "object"
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cy, cx = rng.integers(40, IMG_SIZE - 40, size=2)
    r = rng.integers(20, 50)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r**2
    img[mask] = np.clip(img[mask] + 0.25, 0, 1)
    return np.clip(img, 0, 1)


def _synth_logmel(vibe: int, rng: np.random.Generator) -> np.ndarray:
    """
    Synthetic log-mel: different spectral centroids per vibe.
    Stored as int16 PCM of flattened mel (talex: int16 in TFRecord).
    """
    t = np.linspace(0, 1, AUDIO_FRAMES, dtype=np.float32)
    f = np.linspace(0, 1, N_MELS, dtype=np.float32)
    base_freq = 0.1 + 0.12 * vibe
    grid = np.sin(2 * np.pi * (base_freq + 0.5 * f[None, :]) * (1 + t[:, None]))
    noise = rng.normal(0, 0.15, size=grid.shape).astype(np.float32)
    mel = grid * (0.4 + 0.1 * vibe) + noise
    # Energy envelope
    mel *= (0.5 + 0.5 * np.sin(2 * np.pi * (0.5 + 0.2 * vibe) * t))[:, None]
    mel = np.clip(mel, -3, 3) / 3.0  # ~[-1, 1]
    return mel.astype(np.float32)


def _float_to_int16_pcm(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    q = (x * 32767.0).astype(np.int16)
    return q.tobytes()


def _synth_geo(vibe: int, rng: np.random.Generator) -> np.ndarray:
    # lat_norm, lon_norm, hour_sin, hour_cos, dow_sin, dow_cos, dwell, visit_freq
    hour = float((vibe * 3 + rng.integers(0, 4)) % 24)
    hour_ang = 2 * np.pi * hour / 24.0
    return np.array(
        [
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
            np.sin(hour_ang),
            np.cos(hour_ang),
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
            0.2 + 0.1 * vibe + rng.normal(0, 0.05),
            0.1 + 0.05 * vibe + abs(rng.normal(0, 0.05)),
        ],
        dtype=np.float32,
    )


def _synth_cot(vibe: int, rng: np.random.Generator) -> List[int]:
    # Correlated weak CoT codes
    return [
        int((vibe + rng.integers(0, 2)) % NUM_COT_CLASSES),
        int((vibe * 2 + 1) % NUM_COT_CLASSES),
        int((vibe + 3) % NUM_COT_CLASSES),
        int(vibe % NUM_COT_CLASSES),
    ]


def make_example(
    sample_id: str,
    image: np.ndarray,
    mel: np.ndarray,
    geo: np.ndarray,
    vibe: int,
    cot: List[int],
) -> tf.train.Example:
    # Store image as float32 raw bytes (or JPEG). float32 raw is fine for synth.
    img_bytes = (np.clip(image, 0, 1) * 255.0).astype(np.uint8).tobytes()
    pcm = _float_to_int16_pcm(mel.reshape(-1))
    features = {
        "schema_version": _int64_feature(SCHEMA_VERSION),
        "image": _bytes_feature(img_bytes),
        "audio_pcm": _bytes_feature(pcm),
        "audio_frames": _int64_feature(AUDIO_FRAMES),
        "n_mels": _int64_feature(N_MELS),
        "geo": _float_feature(geo.tolist()),
        "vibe": _int64_feature(int(vibe)),
        "cot": _int64_list_feature(cot),
        "sample_id": _bytes_feature(sample_id.encode("utf-8")),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_split(
    out_dir: Path,
    split: str,
    n: int,
    seed: int = 42,
    shard_size: int = 50,
) -> Manifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + (0 if split == "train" else 1))
    shards: List[str] = []
    row_counts: List[int] = []
    sha: Dict[str, str] = {}
    class_counts: Dict[str, int] = {str(i): 0 for i in range(NUM_VIBE_CLASSES)}

    written = 0
    shard_i = 0
    writer = None
    shard_path = None
    shard_rows = 0

    def _open_shard(i: int):
        nonlocal writer, shard_path, shard_rows
        if writer is not None:
            writer.close()
            sha[shard_path.name] = _sha256_file(shard_path)
            shards.append(shard_path.name)
            row_counts.append(shard_rows)
        shard_path = out_dir / f"{split}-{i:05d}.tfrecord"
        writer = tf.io.TFRecordWriter(str(shard_path))
        shard_rows = 0

    _open_shard(0)
    for i in range(n):
        vibe = int(i % NUM_VIBE_CLASSES)
        class_counts[str(vibe)] += 1
        img = _synth_image(vibe, rng)
        mel = _synth_logmel(vibe, rng)
        geo = _synth_geo(vibe, rng)
        cot = _synth_cot(vibe, rng)
        sid = f"{split}_{i:06d}_v{vibe}"
        ex = make_example(sid, img, mel, geo, vibe, cot)
        writer.write(ex.SerializeToString())
        shard_rows += 1
        written += 1
        if shard_rows >= shard_size and written < n:
            shard_i += 1
            _open_shard(shard_i)

    if writer is not None:
        writer.close()
        sha[shard_path.name] = _sha256_file(shard_path)
        shards.append(shard_path.name)
        row_counts.append(shard_rows)

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        shards=shards,
        row_counts=row_counts,
        class_counts=class_counts,
        sha256=sha,
        split=split,
        vibe_labels=list(VIBE_LABELS),
    )
    man_path = out_dir / f"manifest_{split}.json"
    man_path.write_text(json.dumps(manifest.to_dict(), indent=2))
    print(f"Wrote {written} examples → {out_dir} ({split})")
    print(f"  class_counts={class_counts}")
    print(f"  class_weights={manifest.class_weights_sqrt_inv()}")
    return manifest


def bootstrap(
    out_dir: Path = DEFAULT_TFRECORD_DIR,
    n_train: int = 200,
    n_val: int = 40,
    seed: int = 42,
) -> Tuple[Manifest, Manifest]:
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    m_train = write_split(train_dir, "train", n_train, seed=seed)
    m_val = write_split(val_dir, "val", n_val, seed=seed + 7)
    # Combined pointer
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "train_manifest": str(train_dir / "manifest_train.json"),
        "val_manifest": str(val_dir / "manifest_val.json"),
        "vibe_labels": VIBE_LABELS,
        "note": "Synthetic bootstrap. Replace with PySpark Gold export for real data.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(pointer, indent=2))
    return m_train, m_val


if __name__ == "__main__":
    bootstrap()
