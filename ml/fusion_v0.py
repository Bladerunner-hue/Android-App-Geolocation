"""
Production dense fusion head (v0).

Photo → MobileNetV3Small [576]
Audio → YAMNet mean-pool [1024]
Context → context12-v1 [12]
Mask → [3]  (photo, audio, time=always 1)

  → Dense fusion (~318k params)
  → vibe logits [7] + softmax probs
  → L2 perceptual embedding [128]

No CoT, no KV-cache, no sparse MoE.
Experimental MoE lives under ml/experiments/ (moe_kickstart.py).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

VIBE_LABELS = [
    "serene",
    "energetic",
    "chaotic",
    "nostalgic",
    "tense",
    "social",
    "contemplative",
]
NUM_VIBES = len(VIBE_LABELS)
IMAGE_DIM = 576
AUDIO_DIM = 1024
CONTEXT_DIM = 12
MASK_DIM = 3
PERCEPTUAL_DIM = 128
DEFAULT_HIDDEN = 128  # ~300k params; raise only if capacity-limited
MODEL_SPEC = {
    "name": "fusion_v0",
    "image_dim": IMAGE_DIM,
    "audio_dim": AUDIO_DIM,
    "context_dim": CONTEXT_DIM,
    "mask_dim": MASK_DIM,
    "perceptual_dim": PERCEPTUAL_DIM,
    "num_vibes": NUM_VIBES,
    "vibe_labels": list(VIBE_LABELS),
    "context_revision": "context12-v1",
}


def build_fusion_v0(
    hidden: int = DEFAULT_HIDDEN,
    dropout: float = 0.15,
    name: str = "fusion_v0",
) -> keras.Model:
    """Functional model: feature tensors in → vibe + perceptual out."""
    image_in = keras.Input(shape=(IMAGE_DIM,), name="image_embedding", dtype="float32")
    audio_in = keras.Input(shape=(AUDIO_DIM,), name="audio_embedding", dtype="float32")
    context_in = keras.Input(shape=(CONTEXT_DIM,), name="context", dtype="float32")
    mask_in = keras.Input(shape=(MASK_DIM,), name="modality_mask", dtype="float32")

    # Zero missing modalities (trained condition, not ambiguous zeros alone)
    img = image_in * mask_in[:, 0:1]
    aud = audio_in * mask_in[:, 1:2]
    # time context always present (mask[:,2] == 1); still multiply for consistency
    ctx = context_in * mask_in[:, 2:3]

    img_h = layers.Dense(hidden, activation="gelu", name="img_proj")(img)
    aud_h = layers.Dense(hidden, activation="gelu", name="aud_proj")(aud)
    ctx_h = layers.Dense(hidden // 2, activation="gelu", name="ctx_proj")(ctx)
    mask_h = layers.Dense(16, activation="gelu", name="mask_proj")(mask_in)

    fused = layers.Concatenate(name="fuse_cat")([img_h, aud_h, ctx_h, mask_h])
    x = layers.Dense(hidden, activation="gelu", name="fuse_1")(fused)
    x = layers.Dropout(dropout, name="drop_1")(x)
    x = layers.Dense(hidden, activation="gelu", name="fuse_2")(x)
    x = layers.Dropout(dropout, name="drop_2")(x)

    vibe_logits = layers.Dense(NUM_VIBES, dtype="float32", name="vibe_logits")(x)
    vibe_probabilities = layers.Activation("softmax", dtype="float32", name="vibe_probabilities")(
        vibe_logits
    )
    perc_raw = layers.Dense(PERCEPTUAL_DIM, dtype="float32", name="perceptual_raw")(x)
    perceptual_embedding = layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1),
        name="perceptual_embedding",
    )(perc_raw)
    # Aux classifier on embedding so perceptual_raw receives gradients (not contrastive yet)
    vibe_from_emb = layers.Dense(NUM_VIBES, dtype="float32", name="vibe_from_emb_logits")(
        perceptual_embedding
    )
    vibe_id = layers.Lambda(
        lambda t: tf.argmax(t, axis=-1, output_type=tf.int32),
        name="vibe_id",
    )(vibe_probabilities)

    return keras.Model(
        inputs={
            "image_embedding": image_in,
            "audio_embedding": audio_in,
            "context": context_in,
            "modality_mask": mask_in,
        },
        outputs={
            "vibe_logits": vibe_logits,
            "vibe_probabilities": vibe_probabilities,
            "perceptual_embedding": perceptual_embedding,
            "vibe_from_emb_logits": vibe_from_emb,
            "vibe_id": vibe_id,
        },
        name=name,
    )


def count_trainable_params(model: keras.Model) -> int:
    return int(sum(int(tf.size(v)) for v in model.trainable_variables))


def modality_dropout(
    batch: Dict[str, tf.Tensor],
    rate: float,
    training: bool,
) -> Dict[str, tf.Tensor]:
    """Randomly drop photo or audio at train time (not time context)."""
    if not training or rate <= 0:
        return batch
    b = tf.shape(batch["modality_mask"])[0]
    # drop image
    keep_img = tf.cast(tf.random.uniform([b, 1]) > rate, tf.float32)
    keep_aud = tf.cast(tf.random.uniform([b, 1]) > rate, tf.float32)
    mask = batch["modality_mask"]
    new_mask = tf.concat(
        [mask[:, 0:1] * keep_img, mask[:, 1:2] * keep_aud, mask[:, 2:3]],
        axis=-1,
    )
    out = dict(batch)
    out["modality_mask"] = new_mask
    out["image_embedding"] = batch["image_embedding"] * new_mask[:, 0:1]
    out["audio_embedding"] = batch["audio_embedding"] * new_mask[:, 1:2]
    return out


def dummy_batch(n: int = 4, seed: int = 0) -> Dict[str, tf.Tensor]:
    g = tf.random.Generator.from_seed(seed)
    return {
        "image_embedding": g.normal([n, IMAGE_DIM]),
        "audio_embedding": g.normal([n, AUDIO_DIM]),
        "context": g.normal([n, CONTEXT_DIM]),
        "modality_mask": tf.constant(
            [[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]][:n],
            dtype=tf.float32,
        ),
    }


if __name__ == "__main__":
    m = build_fusion_v0()
    b = dummy_batch(2)
    o = m(b, training=False)
    print("params", count_trainable_params(m))
    print({k: tuple(v.shape) for k, v in o.items()})
