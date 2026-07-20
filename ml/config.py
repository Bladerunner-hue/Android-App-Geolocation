"""Shared training / schema config (single source of truth for fusion_v0 + experimental MoE)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

# Keep in sync with ml/fusion_v0.py and Kotlin FusionV0Interpreter / ContextEncoderV1.
SCHEMA_VERSION = 1
NUM_VIBE_CLASSES = 7
VIBE_LABELS = [
    "serene",
    "energetic",
    "chaotic",
    "nostalgic",
    "tense",
    "social",
    "contemplative",
]
NUM_COT_SLOTS = 4
NUM_COT_CLASSES = 8
IMG_SIZE = 224
N_MELS = 64
AUDIO_FRAMES = 96
GEO_RAW_DIM = 8

# Dense fusion_v0 I/O contract (must match Kotlin TFLite signatures).
IMAGE_EMBED_DIM = 576
AUDIO_EMBED_DIM = 1024
CONTEXT_DIM = 12
MODALITY_MASK_DIM = 3
PERCEPTUAL_DIM = 128

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "ml" / "data_sample"
DEFAULT_TFRECORD_DIR = DEFAULT_DATA_DIR / "tfrecords"
DEFAULT_MANIFEST = DEFAULT_TFRECORD_DIR / "manifest.json"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
DEFAULT_SAVEDMODEL_DIR = REPO_ROOT / "ml" / "saved_models" / "fusion_v0" / "1"
DEFAULT_TFLITE_PATH = REPO_ROOT / "ml" / "exports" / "fusion_v0.tflite"
EXPERIMENTS_DIR = REPO_ROOT / "ml" / "experiments"

# TFRecord feature map for experimental raw-media MoE path (int16 PCM).
# Production fusion_v0 trains from NPZ embeddings via train_fusion_v0.py.
FEATURE_SPEC_KEYS = (
    "schema_version",
    "image",
    "audio_pcm",
    "audio_frames",
    "n_mels",
    "geo",
    "vibe",
    "cot",
    "sample_id",
)


def feature_spec():
    """Lazy TF feature spec so pure-Python importers need not load TensorFlow."""
    import tensorflow as tf

    return {
        "schema_version": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "audio_pcm": tf.io.FixedLenFeature([], tf.string),  # int16 little-endian
        "audio_frames": tf.io.FixedLenFeature([], tf.int64),
        "n_mels": tf.io.FixedLenFeature([], tf.int64),
        "geo": tf.io.FixedLenFeature([GEO_RAW_DIM], tf.float32),
        "vibe": tf.io.FixedLenFeature([], tf.int64),
        "cot": tf.io.FixedLenFeature([NUM_COT_SLOTS], tf.int64),
        "sample_id": tf.io.FixedLenFeature([], tf.string),
    }


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    accum_steps: int = 4
    cot_weight: float = 0.35
    label_smoothing: float = 0.1
    hidden: int = 256
    num_experts: int = 4
    top_k: int = 2
    num_blocks: int = 2
    lora_rank: int = 8
    freeze_vision: bool = True
    mixed_precision: bool = True
    shuffle_buffer: int = 256
    seed: int = 42
    # Tiny bootstrap: synthetic examples if no TFRecords yet
    synthetic_samples: int = 200
    val_fraction: float = 0.15

    def steps_per_epoch(self, n_train: int) -> int:
        return max(n_train // self.batch_size, 1)


@dataclass
class Manifest:
    schema_version: int = SCHEMA_VERSION
    shards: List[str] = field(default_factory=list)
    row_counts: List[int] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)
    sha256: Dict[str, str] = field(default_factory=dict)
    img_size: int = IMG_SIZE
    n_mels: int = N_MELS
    audio_frames: int = AUDIO_FRAMES
    geo_raw_dim: int = GEO_RAW_DIM
    vibe_labels: List[str] = field(default_factory=lambda: list(VIBE_LABELS))
    split: str = "train"  # train | val | test

    def total_rows(self) -> int:
        return int(sum(self.row_counts))

    def class_weights_sqrt_inv(self) -> List[float]:
        """Inverse-sqrt frequency, renormalised to sum = num_classes."""
        counts = [
            float(self.class_counts.get(str(i), self.class_counts.get(VIBE_LABELS[i], 1)))
            for i in range(NUM_VIBE_CLASSES)
        ]
        counts = [c if c > 0 else 1.0 for c in counts]
        inv = [1.0 / (c ** 0.5) for c in counts]
        s = sum(inv)
        target = float(NUM_VIBE_CLASSES)
        return [w * target / s for w in inv]

    def to_dict(self) -> dict:
        return asdict(self)
