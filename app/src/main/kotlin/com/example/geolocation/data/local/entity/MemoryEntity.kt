package com.example.geolocation.data.local.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Offline-first journal row. Sync bookkeeping is local-only.
 * AI telemetry lives in structuredEvidenceJson / perceptual / insight JSON —
 * aligned with backend 002 / fusion_v0.
 */
@Entity(
    tableName = "memories",
    indices = [
        Index(value = ["clientUuid"], unique = true),
        Index(value = ["capturedAtMs"]),
        Index(value = ["syncStatus"]),
    ],
)
data class MemoryEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val clientUuid: String,
    val photoPath: String?,
    val audioPath: String?,
    val latitude: Double?,
    val longitude: Double?,
    val caption: String?,
    val vibeLabel: String?,
    val vibeConfidence: Float?,
    /** unavailable | on_device | pending */
    val analysisStatus: String,
    val privateMode: Boolean,
    val cloudSyncEnabled: Boolean,
    val enrichmentEnabled: Boolean,
    val capturedAtMs: Long,
    val createdAtMs: Long,
    /** pending | synced | failed | not_applicable */
    val syncStatus: String,
    val lastSyncError: String?,
    val serverId: Long?,
    /** @deprecated prefer structuredEvidenceJson */
    val evidenceJson: String? = null,
    val modelVersion: String? = null,
    /** on_device | server_fusion | rules | unavailable */
    val analysisSource: String = "unavailable",
    /** JSON float[128] fusion_v0 perceptual (NOT E5) */
    val perceptualEmbeddingJson: String? = null,
    /** JSON float[128] optional insight (null until aux head ships) */
    val insightEmbeddingJson: String? = null,
    /**
     * JSON float[1024] E5 semantic (intfloat/e5-large-v2).
     * Same space as Postgres memory_semantic_embeddings — never fusion perceptual.
     */
    val semanticEmbeddingJson: String? = null,
    /**
     * vibe_probs[7], modality_mask[3], context12, context12_revision, source, contract
     */
    val structuredEvidenceJson: String? = null,
)
