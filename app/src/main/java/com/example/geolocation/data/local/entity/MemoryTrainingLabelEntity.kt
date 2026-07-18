package com.example.geolocation.data.local.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Train Mode labels — captured BEFORE model prediction is shown
 * to avoid anchoring bias.
 *
 * Consent for training is separate from cloud upload consent.
 * Corrections append new rows rather than overwriting.
 */
@Entity(
    tableName = "memory_training_labels",
    indices = [Index("memoryId"), Index("sessionId")],
)
data class MemoryTrainingLabelEntity(
    @PrimaryKey val id: String,
    val memoryId: String,
    val sessionId: String,
    val primaryVibe: String,
    /** JSON array of optional secondary vibe strings. */
    val secondaryVibesJson: String,
    /** -2..2 or null */
    val valence: Int?,
    /** 1..5 or null */
    val arousal: Int?,
    /** 1..3 */
    val labelConfidence: Int,
    /** human_self | human_reviewed | human_quick */
    val labelSource: String,
    val utcOffsetMinutes: Int,
    val locationAccuracyMeters: Float?,
    val consentForTraining: Boolean,
    val consentForCloud: Boolean,
    val labelledAtEpochMillis: Long,
    /** If this row corrects an earlier label, point at prior id. */
    val correctsLabelId: String? = null,
)
