"""
tf.data pipeline — tensorflow-talex canonical pattern.

list_files → interleave(TFRecordDataset) → map(parse) → [cache]
  → shuffle → cheap_augment → batch → prefetch(AUTOTUNE)

Audio: int16 PCM in TFRecords → dequant float32 in parse.
Augment: Spec-style mel mask, image flip/brightness, geo jitter (no time-stretch).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.config import (  # noqa: E402
    AUDIO_FRAMES,
    GEO_RAW_DIM,
    IMG_SIZE,
    N_MELS,
    NUM_COT_SLOTS,
    SCHEMA_VERSION,
    feature_spec,
)

FEATURE_SPEC = feature_spec()


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def parse_example(serialized: tf.Tensor) -> Dict[str, tf.Tensor]:
    parsed = tf.io.parse_single_example(serialized, FEATURE_SPEC)
    schema = parsed["schema_version"]
    # Loader refuses newer schemas
    tf.debugging.assert_less_equal(
        schema,
        tf.constant(SCHEMA_VERSION, dtype=tf.int64),
        message="TFRecord schema newer than loader SCHEMA_VERSION",
    )

    # Image: uint8 raw HWC
    img = tf.io.decode_raw(parsed["image"], tf.uint8)
    img = tf.reshape(img, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.cast(img, tf.float32) / 255.0

    # Audio int16 → float32 mel [T, F, 1]
    pcm = tf.io.decode_raw(parsed["audio_pcm"], tf.int16)
    mel = tf.cast(pcm, tf.float32) / 32767.0
    mel = tf.reshape(mel, [AUDIO_FRAMES, N_MELS, 1])

    geo = parsed["geo"]
    vibe = tf.cast(parsed["vibe"], tf.int32)
    cot = tf.cast(parsed["cot"], tf.int32)

    return {
        "image": img,
        "audio_mel": mel,
        "geo": geo,
        "vibe": vibe,
        "cot": cot,
    }


def cheap_augment(
    sample: Dict[str, tf.Tensor],
    training: bool = True,
) -> Dict[str, tf.Tensor]:
    if not training:
        return sample

    img = sample["image"]
    mel = sample["audio_mel"]
    geo = sample["geo"]

    # Image
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.12)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)

    # Mel: gain + gaussian noise + time/freq mask (SpecAugment-lite)
    gain = tf.random.uniform([], 0.8, 1.2)
    mel = mel * gain
    mel = mel + tf.random.normal(tf.shape(mel), stddev=0.03)
    # Time mask
    t_w = tf.random.uniform([], 1, 8, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, AUDIO_FRAMES - t_w, dtype=tf.int32)
    indices = tf.range(AUDIO_FRAMES)
    t_keep = (indices < t0) | (indices >= t0 + t_w)
    mel = mel * tf.cast(t_keep, mel.dtype)[:, None, None]
    # Freq mask
    f_w = tf.random.uniform([], 1, 6, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, N_MELS - f_w, dtype=tf.int32)
    f_idx = tf.range(N_MELS)
    f_keep = (f_idx < f0) | (f_idx >= f0 + f_w)
    mel = mel * tf.cast(f_keep, mel.dtype)[None, :, None]
    mel = tf.clip_by_value(mel, -1.0, 1.0)

    # Geo jitter
    geo = geo + tf.random.normal(tf.shape(geo), stddev=0.02)

    sample = dict(sample)
    sample["image"] = img
    sample["audio_mel"] = mel
    sample["geo"] = geo
    return sample


def _pack(
    sample: Dict[str, tf.Tensor],
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    x = {
        "image": sample["image"],
        "audio_mel": sample["audio_mel"],
        "geo": sample["geo"],
    }
    y = {"vibe": sample["vibe"], "cot": sample["cot"]}
    return x, y


def build_dataset(
    manifest_path: Path,
    batch_size: int = 8,
    training: bool = True,
    shuffle_buffer: int = 256,
    seed: int = 42,
    cache: bool = True,
) -> tf.data.Dataset:
    man = load_manifest(manifest_path)
    base = Path(manifest_path).parent
    shard_paths = [str(base / s) for s in man["shards"]]
    if not shard_paths:
        raise FileNotFoundError(f"No shards in {manifest_path}")

    files = tf.data.Dataset.from_tensor_slices(shard_paths)
    if training:
        files = files.shuffle(len(shard_paths), seed=seed, reshuffle_each_iteration=True)

    ds = files.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type=""),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda s: cheap_augment(s, training=True),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def class_weights_from_manifest(manifest_path: Path) -> tf.Tensor:
    man = load_manifest(manifest_path)
    # Support both Manifest dict and pointer
    if "class_counts" not in man:
        raise ValueError("manifest missing class_counts — use split manifest")
    from ml.config import Manifest

    m = Manifest(
        schema_version=man.get("schema_version", SCHEMA_VERSION),
        shards=man.get("shards", []),
        row_counts=man.get("row_counts", []),
        class_counts={str(k): int(v) for k, v in man["class_counts"].items()},
    )
    w = m.class_weights_sqrt_inv()
    return tf.constant(w, dtype=tf.float32)
