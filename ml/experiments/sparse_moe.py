"""
Top-k sparse MoE block — pure TensorFlow / Keras (no PyTorch, no MirroredStrategy).

y = sum_{i in top-k} G(x)_i * E_i(x)

Experts: Dense-GELU-Dense. Gate: Dense → softmax (optionally noisy).
Load-balancing via layer.add_loss (Switch Transformer style).
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_balance_loss(router_probs: tf.Tensor, num_experts: int) -> tf.Tensor:
    """Encourage uniform expert usage. router_probs: [N, E]."""
    density = tf.reduce_mean(router_probs, axis=0)
    importance = tf.reduce_mean(router_probs, axis=0)
    loss = tf.reduce_sum(density * importance) * float(num_experts)
    return tf.cast(loss, tf.float32)


class SparseMoEBlock(layers.Layer):
    """MoE on last axis; input [B, H]. Residual connection included."""

    def __init__(
        self,
        hidden: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_mult: int = 2,
        dropout: float = 0.0,
        noisy_gate: bool = False,
        balance_weight: float = 0.01,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k={top_k} invalid for num_experts={num_experts}")
        self.hidden = hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_mult = expert_mult
        self.dropout_rate = dropout
        self.noisy_gate = noisy_gate
        self.balance_weight = balance_weight

        self.gate = layers.Dense(num_experts, name="gate")
        if noisy_gate:
            self.noise_scale = self.add_weight(
                name="noise_scale",
                shape=(),
                initializer=keras.initializers.Constant(0.1),
                trainable=True,
            )
        self.experts = [
            keras.Sequential(
                [
                    layers.Dense(hidden * expert_mult, activation="gelu"),
                    layers.Dense(hidden),
                ],
                name=f"expert_{i}",
            )
            for i in range(num_experts)
        ]
        self.drop = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        h = self.norm(x)
        logits = self.gate(h)
        if self.noisy_gate and training:
            logits = logits + tf.random.normal(tf.shape(logits)) * self.noise_scale
        router_probs = tf.nn.softmax(logits, axis=-1)

        if training and self.balance_weight > 0:
            self.add_loss(self.balance_weight * load_balance_loss(router_probs, self.num_experts))

        top_v, top_i = tf.nn.top_k(router_probs, k=self.top_k)
        top_v = top_v / (tf.reduce_sum(top_v, axis=-1, keepdims=True) + 1e-9)

        expert_outs = tf.stack([e(h, training=training) for e in self.experts], axis=1)
        n = tf.shape(h)[0]
        batch_idx = tf.tile(tf.range(n)[:, None], [1, self.top_k])
        gather_idx = tf.stack([batch_idx, top_i], axis=-1)
        selected = tf.gather_nd(expert_outs, gather_idx)
        weights = tf.expand_dims(top_v, -1)
        y = tf.reduce_sum(selected * weights, axis=1)
        y = self.drop(y, training=training)
        return y + x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "hidden": self.hidden,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "expert_mult": self.expert_mult,
                "dropout": self.dropout_rate,
                "noisy_gate": self.noisy_gate,
                "balance_weight": self.balance_weight,
            }
        )
        return cfg
