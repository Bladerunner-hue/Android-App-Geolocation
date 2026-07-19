package com.example.geolocation.data.remote.dto

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

@JsonClass(generateAdapter = true)
data class TrainingLabelRequest(
    val id: String,
    @Json(name = "memory_id") val memoryId: String,
    @Json(name = "session_id") val sessionId: String,
    @Json(name = "primary_vibe") val primaryVibe: String,
    @Json(name = "secondary_vibes") val secondaryVibes: List<String> = emptyList(),
    val valence: Int? = null,
    val arousal: Int? = null,
    @Json(name = "label_confidence") val labelConfidence: Int,
    @Json(name = "label_source") val labelSource: String = "human_self",
    @Json(name = "utc_offset_minutes") val utcOffsetMinutes: Int,
    @Json(name = "location_accuracy_m") val locationAccuracyM: Float? = null,
    @Json(name = "consent_for_training") val consentForTraining: Boolean = false,
    @Json(name = "consent_for_cloud") val consentForCloud: Boolean = false,
    @Json(name = "labelled_at") val labelledAt: String? = null,
    @Json(name = "corrects_label_id") val correctsLabelId: String? = null,
)

@JsonClass(generateAdapter = true)
data class TrainingLabelDto(
    val id: String,
    @Json(name = "memory_id") val memoryId: String,
    @Json(name = "user_id") val userId: Int,
    @Json(name = "session_id") val sessionId: String,
    @Json(name = "primary_vibe") val primaryVibe: String,
    @Json(name = "secondary_vibes") val secondaryVibes: List<String> = emptyList(),
    val valence: Int? = null,
    val arousal: Int? = null,
    @Json(name = "label_confidence") val labelConfidence: Int,
    @Json(name = "label_source") val labelSource: String,
    @Json(name = "utc_offset_minutes") val utcOffsetMinutes: Int,
    @Json(name = "location_accuracy_m") val locationAccuracyM: Float? = null,
    @Json(name = "consent_for_training") val consentForTraining: Boolean,
    @Json(name = "consent_for_cloud") val consentForCloud: Boolean,
    @Json(name = "labelled_at") val labelledAt: String? = null,
    @Json(name = "corrects_label_id") val correctsLabelId: String? = null,
    @Json(name = "created_at") val createdAt: String? = null,
)
