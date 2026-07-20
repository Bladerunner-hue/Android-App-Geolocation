"""
fusion_moe_v0 — dense fusion_v0 I/O with top-k sparse MoE FFN.

Same contract as fusion_v0 so Kotlin FusionV0Interpreter / TFLite export stay valid:
  in:  image_emb[576], audio_emb[1024], context[12], modality_mask[3]
  out: vibe_logits[7], vibe_probabilities[7], perceptual_embedding L2[128], …

Dense fusion_v0 remains the release baseline until MoE wins on held-out sessions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ml.fusion_v0 import (
    AUDIO_DIM,
    CONTEXT_DIM,
    IMAGE_DIM,
    MASK_DIM,
    NUM_VIBES,
    PERCEPTUAL_DIM,
    VIBE_LABELS,
    count_trainable_params,
    dummy_batch,
    modality_dropout,
)
from ml.experiments.sparse_moe import SparseMoEBlock

DEFAULT_HIDDEN = 128
DEFAULT_EXPERTS = 4
DEFAULT_TOP_K = 2

MODEL_SPEC = {
    "name": "fusion_moe_v0",
    "image_dim": IMAGE_DIM,
    "audio_dim": AUDIO_DIM,
    "context_dim": CONTEXT_DIM,
    "mask_dim": MASK_DIM,
    "perceptual_dim": PERCEPTUAL_DIM,
    "num_vibes": NUM_VIBES,
    "vibe_labels": list(VIBE_LABELS),
    "context_revision": "context12-v1",
    "num_experts": DEFAULT_EXPERTS,
    "top_k": DEFAULT_TOP_K,
}


def build_fusion_moe_v0(
    hidden: int = DEFAULT_HIDDEN,
    dropout: float = 0.15,
    num_experts: int = DEFAULT_EXPERTS,
    top_k: int = DEFAULT_TOP_K,
    use_attention: bool = True,
    balance_weight: float = 0.01,
    name: str = "fusion_moe_v0",
) -> keras.Model:
    """Same I/O names as build_fusion_v0; MoE only in the middle FFN."""
    image_in = keras.Input(shape=(IMAGE_DIM,), name="image_embedding", dtype="float32")
    audio_in = keras.Input(shape=(AUDIO_DIM,), name="audio_embedding", dtype="float32")
    context_in = keras.Input(shape=(CONTEXT_DIM,), name="context", dtype="float32")
    mask_in = keras.Input(shape=(MASK_DIM,), name="modality_mask", dtype="float32")

    img = image_in * mask_in[:, 0:1]
    aud = audio_in * mask_in[:, 1:2]
    ctx = context_in * mask_in[:, 2:3]

    img_h = layers.Dense(hidden, activation="gelu", name="img_proj")(img)
    aud_h = layers.Dense(hidden, activation="gelu", name="aud_proj")(aud)
    ctx_h = layers.Dense(hidden, activation="gelu", name="ctx_proj")(ctx)

    tokens = layers.Lambda(lambda ts: tf.stack(ts, axis=1), name="stack_tokens")(
        [img_h, aud_h, ctx_h]
    )

    if use_attention:
        tokens_n = layers.LayerNormalization(epsilon=1e-5, name="pre_attn_norm")(tokens)
        attn_out = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=max(hidden // 4, 16),
            name="token_mha",
        )(tokens_n, tokens_n)
        tokens = layers.Add(name="attn_res")([tokens, attn_out])
        tokens = layers.Dropout(dropout, name="attn_drop")(tokens)

    pooled = layers.GlobalAveragePooling1D(name="token_pool")(tokens)
    residual_dense = layers.Dense(hidden, activation="gelu", name="dense_residual")(pooled)

    moe_out = SparseMoEBlock(
        hidden=hidden,
        num_experts=num_experts,
        top_k=top_k,
        expert_mult=2,
        dropout=dropout,
        noisy_gate=True,
        balance_weight=balance_weight,
        name="sparse_moe",
    )(pooled)

    x = layers.Add(name="moe_residual_blend")([moe_out, residual_dense])
    x = layers.Dense(hidden, activation="gelu", name="post_moe")(x)
    x = layers.Dropout(dropout, name="post_drop")(x)

    vibe_logits = layers.Dense(NUM_VIBES, dtype="float32", name="vibe_logits")(x)
    vibe_probabilities = layers.Activation("softmax", dtype="float32", name="vibe_probabilities")(
        vibe_logits
    )
    perc_raw = layers.Dense(PERCEPTUAL_DIM, dtype="float32", name="perceptual_raw")(x)
    perceptual_embedding = layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1),
        name="perceptual_embedding",
    )(perc_raw)
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


if __name__ == "__main__":
    m = build_fusion_moe_v0()
    b = dummy_batch(2)
    o = m(b, training=False)
    print("params", count_trainable_params(m))
    print({k: tuple(v.shape) for k, v in o.items()})
