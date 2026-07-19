package com.example.geolocation.data.remote.dto

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Response from POST /api/memories/analyze and GET memory.
 * structured_evidence / evidence are opaque JSON on the wire; we do not need
 * full parse for sync bookkeeping (serverId + status).
 */
@JsonClass(generateAdapter = true)
data class MemoryDto(
    val id: Long,
    @Json(name = "client_uuid") val clientUuid: String,
    val caption: String? = null,
    @Json(name = "vibe_label") val vibeLabel: String? = null,
    @Json(name = "vibe_confidence") val vibeConfidence: Float? = null,
    @Json(name = "analysis_status") val analysisStatus: String,
    @Json(name = "analysis_source") val analysisSource: String? = "unavailable",
    @Json(name = "model_version") val modelVersion: String? = null,
    val latitude: Double? = null,
    val longitude: Double? = null,
    @Json(name = "private_mode") val privateMode: Boolean = false,
    @Json(name = "enrichment_requested") val enrichmentRequested: Boolean = false,
)

@JsonClass(generateAdapter = true)
data class MemorySearchDto(
    val query: String,
    val mode: String,
    val results: List<MemoryDto>,
)

@JsonClass(generateAdapter = true)
data class VibeProfileDto(
    @Json(name = "total_memories") val totalMemories: Int,
    @Json(name = "vibe_counts") val vibeCounts: Map<String, Int>,
    @Json(name = "top_vibes") val topVibes: List<String>,
)
