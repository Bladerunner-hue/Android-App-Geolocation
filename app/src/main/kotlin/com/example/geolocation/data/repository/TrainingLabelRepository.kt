package com.example.geolocation.data.repository

import com.example.geolocation.data.local.PrivacyPreferences
import com.example.geolocation.data.local.TokenStore
import com.example.geolocation.data.local.dao.MemoryTrainingLabelDao
import com.example.geolocation.data.local.entity.MemoryTrainingLabelEntity
import com.example.geolocation.data.remote.api.TrainingApi
import com.example.geolocation.data.remote.dto.TrainingLabelRequest
import com.example.geolocation.domain.TrainModeSession
import com.example.geolocation.util.SyncScheduler
import java.time.Instant
import java.util.TimeZone
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.first
import org.json.JSONArray
import retrofit2.HttpException

/**
 * Human Train Mode labels (truth for training). Optional cloud outbox when consented.
 */
@Singleton
class TrainingLabelRepository @Inject constructor(
    private val labelDao: MemoryTrainingLabelDao,
    private val trainingApi: TrainingApi,
    private val privacy: PrivacyPreferences,
    private val tokenStore: TokenStore,
    private val session: TrainModeSession,
    private val syncScheduler: SyncScheduler,
) {
    suspend fun submitLabel(
        memoryId: String,
        primaryVibe: String,
        valence: Int?,
        arousal: Int?,
        confidence: Int,
        consentTraining: Boolean,
        consentCloud: Boolean,
        noteIgnored: String? = null,
        secondaryVibes: List<String> = emptyList(),
        locationAccuracyM: Float? = null,
        correctsLabelId: String? = null,
    ): MemoryTrainingLabelEntity {
        val now = System.currentTimeMillis()
        val snap = privacy.snapshot.first()
        val canCloud = consentCloud &&
            !snap.privateMode &&
            snap.cloudSyncEnabled &&
            !tokenStore.currentToken().isNullOrBlank()
        val syncStatus = if (canCloud) "pending" else "not_applicable"

        val entity = MemoryTrainingLabelEntity(
            id = UUID.randomUUID().toString(),
            memoryId = memoryId,
            sessionId = session.current(now),
            primaryVibe = primaryVibe,
            secondaryVibesJson = JSONArray(secondaryVibes).toString(),
            valence = valence,
            arousal = arousal,
            labelConfidence = confidence.coerceIn(1, 3),
            labelSource = "human_self",
            utcOffsetMinutes = TimeZone.getDefault().getOffset(now) / 60_000,
            locationAccuracyMeters = locationAccuracyM,
            consentForTraining = consentTraining,
            consentForCloud = consentCloud,
            labelledAtEpochMillis = now,
            correctsLabelId = correctsLabelId,
            syncStatus = syncStatus,
            lastSyncError = null,
        )
        labelDao.insert(entity)
        if (entity.syncStatus == "pending") {
            syncScheduler.enqueueLabelSync()
        }
        return entity
    }

    suspend fun pendingCloudSync(): List<MemoryTrainingLabelEntity> =
        labelDao.pendingCloudSync()

    suspend fun syncOne(label: MemoryTrainingLabelEntity): Result<MemoryTrainingLabelEntity> {
        if (!label.consentForCloud) {
            return Result.failure(IllegalStateException("No cloud consent"))
        }
        if (tokenStore.currentToken().isNullOrBlank()) {
            return Result.failure(IllegalStateException("Not signed in"))
        }
        return try {
            val secondary = try {
                val arr = JSONArray(label.secondaryVibesJson)
                (0 until arr.length()).map { arr.getString(it) }
            } catch (_: Exception) {
                emptyList()
            }
            trainingApi.createLabel(
                TrainingLabelRequest(
                    id = label.id,
                    memoryId = label.memoryId,
                    sessionId = label.sessionId,
                    primaryVibe = label.primaryVibe,
                    secondaryVibes = secondary,
                    valence = label.valence,
                    arousal = label.arousal,
                    labelConfidence = label.labelConfidence,
                    labelSource = label.labelSource,
                    utcOffsetMinutes = label.utcOffsetMinutes,
                    locationAccuracyM = label.locationAccuracyMeters,
                    consentForTraining = label.consentForTraining,
                    consentForCloud = true,
                    labelledAt = Instant.ofEpochMilli(label.labelledAtEpochMillis).toString(),
                    correctsLabelId = label.correctsLabelId,
                ),
            )
            val updated = label.copy(syncStatus = "synced", lastSyncError = null)
            labelDao.update(updated)
            Result.success(updated)
        } catch (e: HttpException) {
            val failed = label.copy(syncStatus = "failed", lastSyncError = e.message())
            labelDao.update(failed)
            Result.failure(e)
        } catch (e: Exception) {
            val failed = label.copy(syncStatus = "failed", lastSyncError = e.message)
            labelDao.update(failed)
            Result.failure(e)
        }
    }
}
