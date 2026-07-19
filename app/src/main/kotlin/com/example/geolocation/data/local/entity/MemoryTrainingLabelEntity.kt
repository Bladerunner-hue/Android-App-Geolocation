package com.example.geolocation.data.local.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Train Mode labels — captured BEFORE model prediction is shown.
 * Mirrors backend `training_labels` (+ local syncStatus).
 */
@Entity(
    tableName = "memory_training_labels",
    indices = [
        Index("memoryId"),
        Index("sessionId"),
        Index("syncStatus"),
    ],
)
data class MemoryTrainingLabelEntity(
    @PrimaryKey val id: String,
    /** Prefer memory clientUuid for server `memory_id`. */
    val memoryId: String,
    val sessionId: String,
    val primaryVibe: String,
    val secondaryVibesJson: String,
    val valence: Int?,
    val arousal: Int?,
    val labelConfidence: Int,
    val labelSource: String,
    val utcOffsetMinutes: Int,
    val locationAccuracyMeters: Float?,
    val consentForTraining: Boolean,
    val consentForCloud: Boolean,
    val labelledAtEpochMillis: Long,
    val correctsLabelId: String? = null,
    /** pending | synced | failed | not_applicable */
    val syncStatus: String = "not_applicable",
    val lastSyncError: String? = null,
)
