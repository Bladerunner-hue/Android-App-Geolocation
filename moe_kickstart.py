"""
GeoAI Companion — Multimodal Sparse MoE (tensorflow-talex patterns)

Vision (EfficientNet-B0) + ambient audio log-mel + geo features
→ Sparse top-k MoE + explicit CoT auxiliary heads + insight embedding.

Small-data path:
  - Freeze vision backbone (transfer learning)
  - LoRA on dense/MoE experts (trainable params collapse ~95%+)
  - Mixed precision; final logits in float32
  - Class weights + label smoothing from manifest
  - Gradient accumulation for effective large batches on one GPU

Pure TensorFlow only. No tensorflow_addons, no PyTorch, no MirroredStrategy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Constants (aligned with ml/config.py)
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1
NUM_VIBE_CLASSES = 7
VIBE_LABELS = (
    "serene",
    "energetic",
    "chaotic",
    "nostalgic",
    "tense",
    "social",
    "contemplative",
)
# CoT intermediate slots: scene, sound, geo_context, valence
NUM_COT_SLOTS = 4
NUM_COT_CLASSES = 8  # small codebook per slot (object/scene tags)

IMG_SIZE = 224
N_MELS = 64
AUDIO_FRAMES = 96  # ~3–4s of log-mel frames
GEO_DIM = 16
DEFAULT_HIDDEN = 256
DEFAULT_EXPERTS = 4
DEFAULT_TOP_K = 2
DEFAULT_LORA_RANK = 8


# ---------------------------------------------------------------------------
# GPU / precision (single device — no MirroredStrategy)
# ---------------------------------------------------------------------------
def setup_runtime(
    mixed_precision: bool = True,
    force_cpu: bool = False,
) -> None:
    """Single-device setup. Set force_cpu=True when cuDNN/driver is broken."""
    if force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        keras.mixed_precision.set_global_policy("float32")
        return
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if mixed_precision and gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        keras.mixed_precision.set_global_policy("float32")


# ---------------------------------------------------------------------------
# LoRA dense
# ---------------------------------------------------------------------------
class LoRADense(layers.Layer):
    """Frozen dense W plus low-rank ΔW = A @ B * (alpha/rank)."""

    def __init__(
        self,
        units: int,
        rank: int = DEFAULT_LORA_RANK,
        alpha: float = 16.0,
        activation: Optional[str] = None,
        use_bias: bool = True,
        train_base: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.train_base = train_base
        self.base = layers.Dense(
            units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer="he_normal",
            name=f"{self.name}_base" if self.name else "base",
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        in_dim = int(input_shape[-1])
        self.base.build(input_shape)
        self.base.trainable = self.train_base
        self.lora_A = self.add_weight(
            name="lora_A",
            shape=(in_dim, self.rank),
            initializer="he_normal",
            trainable=True,
        )
        self.lora_B = self.add_weight(
            name="lora_B",
            shape=(self.rank, self.units),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        base_out = self.base(inputs)
        # Cast for mixed precision stability
        x = tf.cast(inputs, self.lora_A.dtype)
        delta = tf.matmul(tf.matmul(x, self.lora_A), self.lora_B) * self.scaling
        out = base_out + tf.cast(delta, base_out.dtype)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                "units": self.units,
                "rank": self.rank,
                "alpha": self.alpha,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "train_base": self.train_base,
            }
        )
        return cfg


# ---------------------------------------------------------------------------
# Sparse MoE (top-k routing with correct expert index gather)
# ---------------------------------------------------------------------------
class SparseMoEBlock(layers.Layer):
    """
    y = sum_{i in top-k} G(x)_i * E_i(x)

    Dense expert eval (small num_experts) with sparse top-k weighting.
    Suitable for mobile-sized models; true scatter dispatch is optional later.
    """

    def __init__(
        self,
        num_experts: int = DEFAULT_EXPERTS,
        top_k: int = DEFAULT_TOP_K,
        expert_dim: int = DEFAULT_HIDDEN,
        lora_rank: int = DEFAULT_LORA_RANK,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_dim = expert_dim
        self.lora_rank = lora_rank
        self.dropout_rate = dropout
        self.gate = layers.Dense(num_experts, kernel_initializer="glorot_uniform")
        self.experts = [
            keras.Sequential(
                [
                    LoRADense(expert_dim, rank=lora_rank, activation="gelu"),
                    layers.Dropout(dropout),
                    LoRADense(expert_dim, rank=lora_rank, activation=None),
                ],
                name=f"expert_{i}",
            )
            for i in range(num_experts)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = x
        x_n = self.norm(x)

        gate_logits = self.gate(x_n)  # [B, E]
        top_logits, top_idx = tf.nn.top_k(gate_logits, k=self.top_k)  # [B, K]
        gate_vals = tf.nn.softmax(tf.cast(top_logits, tf.float32), axis=-1)
        gate_vals = tf.cast(gate_vals, x.dtype)

        # Stack expert outputs: [B, E, D]
        stacked = tf.stack(
            [expert(x_n, training=training) for expert in self.experts],
            axis=1,
        )

        # Gather top-k experts: [B, K, D]
        batch = tf.shape(x)[0]
        batch_idx = tf.tile(tf.range(batch)[:, None], [1, self.top_k])
        gather_idx = tf.stack([batch_idx, top_idx], axis=-1)
        selected = tf.gather_nd(stacked, gather_idx)

        # Weighted sum: [B, D]
        combined = tf.reduce_sum(selected * gate_vals[:, :, None], axis=1)
        combined = self.drop(combined, training=training)
        return residual + combined

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "expert_dim": self.expert_dim,
                "lora_rank": self.lora_rank,
                "dropout": self.dropout_rate,
            }
        )
        return cfg


# ---------------------------------------------------------------------------
# Pre-norm transformer block (attention + MoE FFN)
# ---------------------------------------------------------------------------
class MoETransformerBlock(layers.Layer):
    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN,
        num_heads: int = 4,
        num_experts: int = DEFAULT_EXPERTS,
        top_k: int = DEFAULT_TOP_K,
        lora_rank: int = DEFAULT_LORA_RANK,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.num_heads = num_heads
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden // num_heads,
            dropout=dropout,
        )
        self.drop1 = layers.Dropout(dropout)
        self.moe = SparseMoEBlock(
            num_experts=num_experts,
            top_k=top_k,
            expert_dim=hidden,
            lora_rank=lora_rank,
            dropout=dropout,
        )

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
        cache_k: Optional[tf.Tensor] = None,
        cache_v: Optional[tf.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        """
        x: [B, T, D]. For fused single-token path T=1.
        Optional KV cache hooks for future autoregressive journal tokens.
        """
        h = self.ln1(x)
        # MHA with optional external key/value (KV-cache path)
        if use_cache and cache_k is not None and cache_v is not None:
            # Append current projected tokens — simplified: recompute attn on concat
            # Full production cache stores projected K/V; here we expose the API.
            attn_out = self.attn(h, h, training=training)
            new_k, new_v = cache_k, cache_v
        else:
            attn_out = self.attn(h, h, training=training)
            new_k, new_v = None, None

        x = x + self.drop1(attn_out, training=training)
        # MoE expects [B, D] when T=1, or apply per-token
        shape = tf.shape(x)
        b, t, d = shape[0], shape[1], shape[2]
        flat = tf.reshape(x, [-1, d])
        flat = self.moe(flat, training=training)
        x = tf.reshape(flat, [b, t, d])
        return x, new_k, new_v


# ---------------------------------------------------------------------------
# Geo sinusoidal embedding
# ---------------------------------------------------------------------------
def geo_sinusoidal_features(geo: tf.Tensor, out_dim: int = GEO_DIM) -> tf.Tensor:
    """
    geo: [B, F] raw features (lat_norm, lon_norm, hour_sin, hour_cos, ...).
    Expand with sin/cos frequency bases for better small-data generalisation.
    """
    geo = tf.cast(geo, tf.float32)
    # Project to half frequencies
    half = out_dim // 2
    # Learnable scale via dense outside; here fixed Fourier features
    freqs = tf.constant(
        [2.0 ** i for i in range(half)], dtype=tf.float32
    )  # [half]
    # Use first feature dim as phase driver, then mix
    # geo already multi-dim: linear mix then sin/cos
    w = tf.linspace(0.0, 1.0, tf.shape(geo)[-1])
    # Simple: dense-free — tile and modulate
    # Reduce geo to scalar mix
    g = tf.reduce_mean(geo, axis=-1, keepdims=True)  # [B, 1]
    angles = g * freqs[None, :] * tf.constant(3.14159265, dtype=tf.float32)
    return tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)  # [B, out_dim]


# ---------------------------------------------------------------------------
# Full model (Functional API builder + subclass for multi-output training)
# ---------------------------------------------------------------------------
class GeoAIMoE(keras.Model):
    """
    Multimodal GeoAI MoE with CoT multi-task heads.

    Inputs (dict or tuple):
      image:      [B, 224, 224, 3] float32 in [0, 1]  (EfficientNet preprocess inside)
      audio_mel:  [B, AUDIO_FRAMES, N_MELS, 1] float32 log-mel
      geo:        [B, GEO_RAW_DIM] float32 engineered features

    Outputs (dict):
      vibe_logits:     [B, 7] float32
      cot_logits:      [B, NUM_COT_SLOTS, NUM_COT_CLASSES] float32
      insight_embedding:[B, hidden] float32 L2-normalised (retrieval / journal)
      gate_aux:        optional load-balance signal
    """

    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN,
        num_experts: int = DEFAULT_EXPERTS,
        top_k: int = DEFAULT_TOP_K,
        num_blocks: int = 2,
        num_heads: int = 4,
        lora_rank: int = DEFAULT_LORA_RANK,
        freeze_vision: bool = True,
        dropout: float = 0.15,
        geo_raw_dim: int = 8,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.geo_raw_dim = geo_raw_dim
        self.freeze_vision = freeze_vision

        # Vision backbone — transfer learning (small data king)
        self.vision = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        self.vision.trainable = not freeze_vision
        self.vision_proj = LoRADense(hidden, rank=lora_rank, activation="gelu")

        # Audio encoder (log-mel CNN — talex voice-emotion style, place-vibe)
        self.audio_encoder = keras.Sequential(
            [
                layers.Input(shape=(AUDIO_FRAMES, N_MELS, 1)),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                layers.Conv2D(128, 3, padding="same", activation="relu"),
                layers.GlobalAveragePooling2D(),
                LoRADense(hidden, rank=lora_rank, activation="gelu"),
            ],
            name="audio_encoder",
        )

        self.geo_dense = LoRADense(hidden, rank=lora_rank, activation="gelu")
        self.fusion = LoRADense(hidden, rank=lora_rank, activation="gelu")
        self.token_drop = layers.Dropout(dropout)

        self.blocks = [
            MoETransformerBlock(
                hidden=hidden,
                num_heads=num_heads,
                num_experts=num_experts,
                top_k=top_k,
                lora_rank=lora_rank,
                dropout=dropout,
                name=f"moe_block_{i}",
            )
            for i in range(num_blocks)
        ]
        self.final_ln = layers.LayerNormalization(epsilon=1e-6)

        # Heads — logits always float32 under mixed precision
        self.vibe_head = layers.Dense(
            NUM_VIBE_CLASSES,
            dtype="float32",
            kernel_initializer="glorot_uniform",
            name="vibe_logits",
        )
        self.cot_head = layers.Dense(
            NUM_COT_SLOTS * NUM_COT_CLASSES,
            dtype="float32",
            kernel_initializer="glorot_uniform",
            name="cot_logits_flat",
        )
        self.insight_head = layers.Dense(
            hidden,
            dtype="float32",
            kernel_initializer="he_normal",
            name="insight_embedding",
        )

    def encode(
        self,
        image: tf.Tensor,
        audio_mel: tf.Tensor,
        geo: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        # EfficientNet expects 0–255 with internal preprocess; scale from [0,1]
        img = tf.cast(image, tf.float32) * 255.0
        img = keras.applications.efficientnet.preprocess_input(img)
        v = self.vision(img, training=training and (not self.freeze_vision))
        v = self.vision_proj(v)

        a = self.audio_encoder(audio_mel, training=training)

        g_fourier = geo_sinusoidal_features(geo, out_dim=GEO_DIM)
        # Pad/concat raw geo + fourier
        g = tf.concat([tf.cast(geo, tf.float32), g_fourier], axis=-1)
        g = self.geo_dense(g)

        fused = self.fusion(tf.concat([v, a, g], axis=-1))
        # Token sequence: [vision_tok, audio_tok, geo_tok, fused_tok]
        tokens = tf.stack([v, a, g, fused], axis=1)  # [B, 4, H]
        tokens = self.token_drop(tokens, training=training)

        for block in self.blocks:
            tokens, _, _ = block(tokens, training=training)
        tokens = self.final_ln(tokens)
        # Pool tokens
        return tf.reduce_mean(tokens, axis=1)

    def call(
        self,
        inputs: Any,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        if isinstance(inputs, dict):
            image = inputs["image"]
            audio_mel = inputs["audio_mel"]
            geo = inputs["geo"]
        else:
            image, audio_mel, geo = inputs

        x = self.encode(image, audio_mel, geo, training=training)

        vibe_logits = self.vibe_head(x)
        cot_flat = self.cot_head(x)
        cot_logits = tf.reshape(
            cot_flat, [-1, NUM_COT_SLOTS, NUM_COT_CLASSES]
        )
        insight = self.insight_head(x)
        insight = tf.nn.l2_normalize(tf.cast(insight, tf.float32), axis=-1)

        return {
            "vibe_logits": vibe_logits,
            "cot_logits": cot_logits,
            "insight_embedding": insight,
            "pooled": tf.cast(x, tf.float32),
        }

    @tf.function(reduce_retracing=True)
    def infer(
        self,
        image: tf.Tensor,
        audio_mel: tf.Tensor,
        geo: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        out = self(
            {"image": image, "audio_mel": audio_mel, "geo": geo},
            training=False,
        )
        vibe_prob = tf.nn.softmax(out["vibe_logits"], axis=-1)
        vibe_id = tf.argmax(vibe_prob, axis=-1)
        cot_ids = tf.argmax(out["cot_logits"], axis=-1)
        return {
            "vibe_id": vibe_id,
            "vibe_prob": vibe_prob,
            "cot_ids": cot_ids,
            "insight_embedding": out["insight_embedding"],
        }

    def unfreeze_vision_top(self, n_blocks: int = 20) -> None:
        """Gradual unfreeze for stage-2 fine-tune on small personal data."""
        self.vision.trainable = True
        for layer in self.vision.layers[:-n_blocks]:
            layer.trainable = False
        self.freeze_vision = False


def build_geoai_moe(
    hidden: int = DEFAULT_HIDDEN,
    num_experts: int = DEFAULT_EXPERTS,
    top_k: int = DEFAULT_TOP_K,
    num_blocks: int = 2,
    freeze_vision: bool = True,
    lora_rank: int = DEFAULT_LORA_RANK,
    geo_raw_dim: int = 8,
) -> GeoAIMoE:
    model = GeoAIMoE(
        hidden=hidden,
        num_experts=num_experts,
        top_k=top_k,
        num_blocks=num_blocks,
        freeze_vision=freeze_vision,
        lora_rank=lora_rank,
        geo_raw_dim=geo_raw_dim,
    )
    # Build once with dummy shapes
    _ = model(
        {
            "image": tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]),
            "audio_mel": tf.zeros([1, AUDIO_FRAMES, N_MELS, 1]),
            "geo": tf.zeros([1, geo_raw_dim]),
        },
        training=False,
    )
    return model


# ---------------------------------------------------------------------------
# Losses (multi-task + optional load balance)
# ---------------------------------------------------------------------------
def vibe_loss_fn(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    class_weights: Optional[tf.Tensor] = None,
    label_smoothing: float = 0.1,
) -> tf.Tensor:
    """Sparse labels + optional smoothing via one-hot (portable across TF versions)."""
    y_true = tf.cast(y_true, tf.int32)
    n_class = tf.shape(y_pred)[-1]
    y_oh = tf.one_hot(y_true, depth=n_class, dtype=tf.float32)
    if label_smoothing and label_smoothing > 0:
        y_oh = y_oh * (1.0 - label_smoothing) + (label_smoothing / tf.cast(n_class, tf.float32))
    loss = keras.losses.categorical_crossentropy(
        y_oh, tf.cast(y_pred, tf.float32), from_logits=True
    )
    if class_weights is not None:
        w = tf.gather(class_weights, y_true)
        loss = loss * tf.cast(w, loss.dtype)
    return tf.reduce_mean(loss)


def cot_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """y_true: [B, NUM_COT_SLOTS] int ids; y_pred: [B, NUM_COT_SLOTS, C]."""
    y_true = tf.cast(y_true, tf.int32)
    n_class = tf.shape(y_pred)[-1]
    y_oh = tf.one_hot(y_true, depth=n_class, dtype=tf.float32)
    loss = keras.losses.categorical_crossentropy(
        y_oh, tf.cast(y_pred, tf.float32), from_logits=True
    )
    return tf.reduce_mean(loss)


def total_train_loss(
    y_vibe: tf.Tensor,
    y_cot: tf.Tensor,
    outputs: Dict[str, tf.Tensor],
    class_weights: Optional[tf.Tensor] = None,
    cot_weight: float = 0.35,
    label_smoothing: float = 0.1,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    lv = vibe_loss_fn(
        y_vibe,
        outputs["vibe_logits"],
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )
    lc = cot_loss_fn(y_cot, outputs["cot_logits"])
    total = lv + cot_weight * lc
    return total, {"vibe_loss": lv, "cot_loss": lc, "total_loss": total}


# ---------------------------------------------------------------------------
# Gradient accumulation train step
# ---------------------------------------------------------------------------
class GradAccumTrainer:
    def __init__(
        self,
        model: GeoAIMoE,
        optimizer: keras.optimizers.Optimizer,
        class_weights: Optional[tf.Tensor] = None,
        accum_steps: int = 4,
        cot_weight: float = 0.35,
        label_smoothing: float = 0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.class_weights = class_weights
        self.accum_steps = max(1, accum_steps)
        self.cot_weight = cot_weight
        self.label_smoothing = label_smoothing
        self._step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._grad_accum: List[tf.Variable] = []
        self._built = False

    def _ensure_accum(self) -> None:
        if self._built:
            return
        self._grad_accum = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in self.model.trainable_variables
        ]
        self._built = True

    @tf.function(reduce_retracing=True)
    def train_step(
        self,
        image: tf.Tensor,
        audio_mel: tf.Tensor,
        geo: tf.Tensor,
        y_vibe: tf.Tensor,
        y_cot: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        self._ensure_accum()
        with tf.GradientTape() as tape:
            outputs = self.model(
                {"image": image, "audio_mel": audio_mel, "geo": geo},
                training=True,
            )
            loss, parts = total_train_loss(
                y_vibe,
                y_cot,
                outputs,
                class_weights=self.class_weights,
                cot_weight=self.cot_weight,
                label_smoothing=self.label_smoothing,
            )
            # Scale for accumulation
            scaled = loss / tf.cast(self.accum_steps, loss.dtype)

        grads = tape.gradient(scaled, self.model.trainable_variables)
        for acc, g in zip(self._grad_accum, grads):
            if g is not None:
                acc.assign_add(g)

        self._step.assign_add(1)
        apply = tf.equal(
            tf.math.floormod(self._step, self.accum_steps), 0
        )

        def _apply() -> tf.Tensor:
            self.optimizer.apply_gradients(
                zip(self._grad_accum, self.model.trainable_variables)
            )
            for acc in self._grad_accum:
                acc.assign(tf.zeros_like(acc))
            return tf.constant(1.0)

        def _skip() -> tf.Tensor:
            return tf.constant(0.0)

        applied = tf.cond(apply, _apply, _skip)
        parts = {k: v for k, v in parts.items()}
        parts["applied"] = applied
        return parts


def make_optimizer(
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    total_steps: int = 2000,
) -> keras.optimizers.Optimizer:
    """AdamW + warmup + cosine decay (talex schedule)."""
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=max(total_steps - warmup_steps, 1),
        alpha=0.05,
    )
    # Linear warmup via custom schedule
    class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base, warmup, peak_lr):
            self.base = base
            self.warmup = warmup
            self.peak_lr = peak_lr

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warm = self.peak_lr * (step + 1.0) / float(max(self.warmup, 1))
            cos = self.base(tf.maximum(step - float(self.warmup), 0.0))
            return tf.where(step < float(self.warmup), warm, cos)

        def get_config(self):
            return {"peak_lr": self.peak_lr, "warmup": self.warmup}

    lr_sched = WarmupCosine(schedule, warmup_steps, lr)
    try:
        opt = keras.optimizers.AdamW(
            learning_rate=lr_sched, weight_decay=weight_decay
        )
    except AttributeError:
        opt = keras.optimizers.Adam(learning_rate=lr_sched)
    # Loss scale for mixed float16
    if keras.mixed_precision.global_policy().name == "mixed_float16":
        opt = keras.mixed_precision.LossScaleOptimizer(opt)
    return opt


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def _smoke() -> None:
    # Prefer CPU for portable smoke (cuDNN/driver mismatches are common)
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    setup_runtime(mixed_precision=False, force_cpu=True)
    model = build_geoai_moe(hidden=128, num_experts=4, num_blocks=1, lora_rank=4)
    trainable = sum(
        int(tf.size(v)) for v in model.trainable_variables
    )
    total = sum(int(tf.size(v)) for v in model.variables)
    print(f"GeoAIMoE built. trainable={trainable:,} / total={total:,}")

    b = 2
    image = tf.random.uniform([b, IMG_SIZE, IMG_SIZE, 3])
    audio = tf.random.normal([b, AUDIO_FRAMES, N_MELS, 1])
    geo = tf.random.normal([b, 8])
    y_vibe = tf.constant([0, 3], dtype=tf.int32)
    y_cot = tf.constant([[1, 2, 0, 3], [0, 1, 2, 1]], dtype=tf.int32)

    out = model({"image": image, "audio_mel": audio, "geo": geo}, training=True)
    assert out["vibe_logits"].shape == (b, NUM_VIBE_CLASSES)
    assert out["cot_logits"].shape == (b, NUM_COT_SLOTS, NUM_COT_CLASSES)

    opt = make_optimizer(lr=1e-3, total_steps=50, warmup_steps=5)
    trainer = GradAccumTrainer(model, opt, accum_steps=2)
    metrics = trainer.train_step(image, audio, geo, y_vibe, y_cot)
    print("train_step metrics:", {k: float(v) for k, v in metrics.items()})
    pred = model.infer(image, audio, geo)
    print("infer vibe_id:", pred["vibe_id"].numpy())
    print("SMOKE OK")


if __name__ == "__main__":
    _smoke()
