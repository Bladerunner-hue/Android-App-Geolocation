package com.example.geolocation.data.local.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Offline-first journal row. Sync state is local until WorkManager succeeds.
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
    val analysisStatus: String, // unavailable | on_device | pending_sync | synced
    val privateMode: Boolean,
    val cloudSyncEnabled: Boolean,
    val enrichmentEnabled: Boolean,
    val capturedAtMs: Long,
    val createdAtMs: Long,
    val syncStatus: String, // pending | synced | failed | not_applicable
    val lastSyncError: String?,
    val serverId: Long?,
    val evidenceJson: String?,
)
