"""
Frozen feature extractors for fusion_v0.

- MobileNetV3Small ImageNet, pool=avg → [576], pixels [0,255], include_preprocessing=True
- YAMNet TF-Hub → mean frame embeddings [1024], mono 16 kHz float32 waveform

Do not loudness-normalize every clip. Preserve amplitude.
Canonicalize EXIF orientation before hashing/extraction (caller responsibility).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

IMAGE_DIM = 576
AUDIO_DIM = 1024
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ImageEncoder:
    """MobileNetV3Small → 576-D."""

    def __init__(self) -> None:
        self.model = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling="avg",
            weights="imagenet",
            include_preprocessing=True,
        )
        self.model.trainable = False
        self.revision = "keras/MobileNetV3Small-ImageNet-224-pool-avg"

    @tf.function
    def embed_jpeg(self, jpeg_bytes: tf.Tensor) -> tf.Tensor:
        image = tf.io.decode_jpeg(jpeg_bytes, channels=3)
        image = tf.cast(image, tf.float32)
        shape = tf.shape(image)
        side = tf.minimum(shape[0], shape[1])
        top = (shape[0] - side) // 2
        left = (shape[1] - side) // 2
        image = tf.image.crop_to_bounding_box(image, top, left, side, side)
        image = tf.image.resize(image, [224, 224], antialias=True)
        vector = self.model(image[None, ...], training=False)[0]
        return tf.ensure_shape(vector, [IMAGE_DIM])

    def embed_jpeg_bytes(self, data: bytes) -> np.ndarray:
        return self.embed_jpeg(tf.constant(data)).numpy().astype(np.float32)

    def embed_path(self, path: Path) -> np.ndarray:
        return self.embed_jpeg_bytes(Path(path).read_bytes())


class AudioEncoder:
    """YAMNet mean-pool → 1024-D. Requires tensorflow_hub."""

    def __init__(self, handle: str = YAMNET_HANDLE) -> None:
        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise ImportError(
                "tensorflow-hub is required for YAMNet. "
                "pip install tensorflow-hub"
            ) from exc
        self.yamnet = hub.load(handle)
        self.handle = handle
        self.revision = "tfhub/google/yamnet/1-mean-pool"

    @tf.function
    def embed_waveform(self, waveform: tf.Tensor) -> tf.Tensor:
        """waveform: float32 mono in [-1, 1], any length."""
        waveform = tf.reshape(waveform, [-1])
        waveform = tf.clip_by_value(tf.cast(waveform, tf.float32), -1.0, 1.0)
        _scores, frame_embeddings, _spec = self.yamnet(waveform)
        vector = tf.reduce_mean(frame_embeddings, axis=0)
        return tf.ensure_shape(vector, [AUDIO_DIM])

    def embed_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
        waveform, sample_rate = tf.audio.decode_wav(
            tf.constant(wav_bytes), desired_channels=1
        )
        sr = int(sample_rate.numpy())
        if sr != 16_000:
            raise ValueError(f"Expected 16 kHz WAV, got {sr}. Resample before embedding.")
        wf = tf.squeeze(waveform, axis=-1)
        return self.embed_waveform(wf).numpy().astype(np.float32)

    def embed_path(self, path: Path) -> np.ndarray:
        return self.embed_wav_bytes(Path(path).read_bytes())

    def embed_pcm_float(self, samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        if sample_rate != 16_000:
            raise ValueError("sample_rate must be 16000")
        wf = tf.constant(samples.astype(np.float32).reshape(-1))
        return self.embed_waveform(wf).numpy().astype(np.float32)


def load_wav_mono_16k(path: Path) -> np.ndarray:
    """Load audio via soundfile if available, else tf.audio (WAV only)."""
    path = Path(path)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if sr != 16_000:
            # simple resample via scipy if needed
            try:
                from scipy import signal

                n = int(len(data) * 16_000 / sr)
                data = signal.resample(data, n)
            except Exception as exc:
                raise ValueError(f"Need 16 kHz audio; got {sr}") from exc
        return np.clip(data.astype(np.float32), -1.0, 1.0)
    except ImportError:
        wav_bytes = path.read_bytes()
        waveform, sample_rate = tf.audio.decode_wav(wav_bytes, desired_channels=1)
        if int(sample_rate.numpy()) != 16_000:
            raise ValueError("WAV must be 16 kHz without soundfile/scipy resample")
        return tf.squeeze(waveform, axis=-1).numpy().astype(np.float32)


def zero_image() -> np.ndarray:
    return np.zeros((IMAGE_DIM,), dtype=np.float32)


def zero_audio() -> np.ndarray:
    return np.zeros((AUDIO_DIM,), dtype=np.float32)
