package com.example.geolocation.data.local.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Offline-first journal row. Sync bookkeeping is local-only.
 * AI telemetry (probs, perceptual, context12) lives in structuredEvidenceJson
 * and perceptualEmbeddingJson — aligned with backend 002 / fusion_v0.
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
    /** fusion_v0, etc. */
    val modelVersion: String? = null,
    /** on_device | server_fusion | rules | unavailable */
    val analysisSource: String = "unavailable",
    /** JSON float[128] perceptual embedding from fusion_v0 */
    val perceptualEmbeddingJson: String? = null,
    /**
     * Production evidence blob:
     * vibe_probs[7], modality_mask[3], context12, context12_revision, source, contract
     */
    val structuredEvidenceJson: String? = null,
)
