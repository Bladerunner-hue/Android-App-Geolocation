package com.example.geolocation.data.repository

import com.example.geolocation.data.local.PrivacyPreferences
import com.example.geolocation.data.local.TokenStore
import com.example.geolocation.data.local.dao.MemoryDao
import com.example.geolocation.data.local.entity.MemoryEntity
import com.example.geolocation.data.ml.AmbientAudioRecorder
import com.example.geolocation.data.ml.ContextEncoderV1
import com.example.geolocation.data.ml.EdgeAnalysisResult
import com.example.geolocation.data.ml.EdgeMemoryAnalyzer
import com.example.geolocation.data.ml.MemoryAnalysisInput
import com.example.geolocation.data.remote.api.MemoryApi
import com.example.geolocation.data.remote.mapper.MemoryAnalyzeMapper
import com.example.geolocation.data.telemetry.HiddenTelemetryCollector
import com.example.geolocation.data.telemetry.TelemetryPipelineFeeder
import com.example.geolocation.util.SyncScheduler
import java.io.File
import java.util.TimeZone
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import retrofit2.HttpException

/**
 * Offline-first capture + outbox sync.
 * Edge computes optional fusion telemetry; server stores — never invents vibes.
 */
@Singleton
class MemoryRepository @Inject constructor(
    private val memoryDao: MemoryDao,
    private val privacy: PrivacyPreferences,
    private val edgeAnalyzer: EdgeMemoryAnalyzer,
    private val audioRecorder: AmbientAudioRecorder,
    private val memoryApi: MemoryApi,
    private val syncScheduler: SyncScheduler,
    private val tokenStore: TokenStore,
    private val telemetry: HiddenTelemetryCollector,
    private val telemetryFeeder: TelemetryPipelineFeeder,
) {
    fun observeMemories(): Flow<List<MemoryEntity>> = memoryDao.observeAll()

    suspend fun searchLocal(q: String): List<MemoryEntity> = memoryDao.search(q.trim())

    suspend fun capture(
        photoFile: File?,
        recordAudio: Boolean,
        audioOut: File?,
        latitude: Double?,
        longitude: Double?,
        caption: String?,
        locationAccuracyM: Float? = null,
    ): MemoryEntity {
        val snap = privacy.snapshot.first()
        var audioPath: String? = null
        if (recordAudio && snap.audioCaptureEnabled && audioOut != null) {
            val result = audioRecorder.recordToFile(audioOut)
            audioPath = result?.file?.absolutePath
        }

        val now = System.currentTimeMillis()
        val offsetMin = TimeZone.getDefault().getOffset(now) / 60_000
        val context12 = ContextEncoderV1.encode(
            epochMillisUtc = now,
            utcOffsetMinutes = offsetMin,
            latitude = latitude,
            longitude = longitude,
            // Never pass 0.0 as "unknown" — encoder treats null as missing accuracy.
            accuracyM = locationAccuracyM,
        )
        // Canonical mask: [photo, audio, time=1]. Location is context12 feature 11 only.
        val modalityMask = ContextEncoderV1.modalityMask(
            photoPresent = photoFile != null,
            audioPresent = audioPath != null,
        )

        var vibe: String? = null
        var conf: Float? = null
        var probs: FloatArray? = null
        var perceptual: FloatArray? = null
        var modelVersion: String? = null
        var analysisSource = "unavailable"
        var analysisStatus = "unavailable"
        var lastError: String? = null

        when (
            val edge = edgeAnalyzer.analyze(
                MemoryAnalysisInput(
                    photo = photoFile,
                    audio = audioPath?.let { File(it) },
                    hasLocation = latitude != null && longitude != null,
                    context12 = context12,
                    modalityMask = modalityMask,
                ),
            )
        ) {
            is EdgeAnalysisResult.Success -> {
                vibe = edge.vibeLabel
                conf = edge.confidence
                probs = edge.probabilities
                perceptual = edge.perceptualEmbedding
                modelVersion = "${edge.model.modelId}@${edge.model.revision}"
                analysisSource = "on_device"
                analysisStatus = "on_device"
            }
            is EdgeAnalysisResult.Unavailable -> lastError = edge.reason
            is EdgeAnalysisResult.Failed -> lastError = "${edge.code}: ${edge.detail}"
        }

        val structured = MemoryAnalyzeMapper.buildStructuredEvidence(
            vibeProbs = probs,
            context12 = context12,
            modalityMask = modalityMask,
            source = analysisSource,
        )

        val syncStatus = when {
            snap.privateMode -> "not_applicable"
            snap.cloudSyncEnabled -> "pending"
            else -> "not_applicable"
        }

        val entity = MemoryEntity(
            clientUuid = UUID.randomUUID().toString(),
            photoPath = photoFile?.absolutePath,
            audioPath = audioPath,
            latitude = latitude,
            longitude = longitude,
            caption = caption,
            vibeLabel = vibe,
            vibeConfidence = conf,
            analysisStatus = analysisStatus,
            privateMode = snap.privateMode,
            cloudSyncEnabled = snap.cloudSyncEnabled && !snap.privateMode,
            enrichmentEnabled = snap.enrichmentEnabled && snap.cloudSyncEnabled && !snap.privateMode,
            capturedAtMs = now,
            createdAtMs = now,
            syncStatus = syncStatus,
            lastSyncError = lastError,
            serverId = null,
            evidenceJson = null,
            modelVersion = modelVersion,
            analysisSource = analysisSource,
            perceptualEmbeddingJson = perceptual?.let { MemoryAnalyzeMapper.floatsToJson(it) },
            insightEmbeddingJson = null,
            // E5 1024-D is filled by server backfill / optional client HTTP — not fusion.
            semanticEmbeddingJson = null,
            structuredEvidenceJson = structured,
        )
        val id = memoryDao.insert(entity)
        val saved = entity.copy(id = id)
        if (saved.syncStatus == "pending") {
            syncScheduler.enqueueMemorySync()
        }
        // Hidden telemetry: silently record capture event regardless of privacy
        telemetry.onMemoryCaptured(saved)
        // Trigger pipeline feed for seamless data flow
        telemetryFeeder.feed()
        return saved
    }

    suspend fun deleteLocal(id: Long) {
        memoryDao.deleteById(id)
    }

    suspend fun syncOne(memory: MemoryEntity): Result<MemoryEntity> {
        if (memory.privateMode || !memory.cloudSyncEnabled) {
            return Result.failure(IllegalStateException("Private or sync disabled"))
        }
        if (tokenStore.currentToken().isNullOrBlank()) {
            return Result.failure(IllegalStateException("Not signed in"))
        }
        return try {
            val parts = MemoryAnalyzeMapper.toParts(memory)
            val dto = memoryApi.analyze(
                clientUuid = parts.clientUuid,
                privateMode = parts.privateMode,
                caption = parts.caption,
                latitude = parts.latitude,
                longitude = parts.longitude,
                capturedAt = parts.capturedAt,
                onDeviceVibe = parts.onDeviceVibe,
                onDeviceConfidence = parts.onDeviceConfidence,
                onDeviceProbs = parts.onDeviceProbs,
                perceptualEmbedding = parts.perceptualEmbedding,
                insightEmbedding = parts.insightEmbedding,
                modelVersion = parts.modelVersion,
                analysisSource = parts.analysisSource,
                structuredEvidence = parts.structuredEvidence,
                requestEnrichment = parts.requestEnrichment,
                photo = parts.photo,
                audio = parts.audio,
            )
            val updated = memory.copy(
                serverId = dto.id,
                syncStatus = "synced",
                lastSyncError = null,
                analysisStatus = dto.analysisStatus,
                analysisSource = dto.analysisSource ?: memory.analysisSource,
                modelVersion = dto.modelVersion ?: memory.modelVersion,
                vibeLabel = dto.vibeLabel ?: memory.vibeLabel,
            )
            memoryDao.update(updated)
            Result.success(updated)
        } catch (e: HttpException) {
            val msg = when (e.code()) {
                401 -> "Unauthorized — sign in again"
                403 -> "Forbidden (private mode or consent)"
                else -> e.message()
            }
            val failed = memory.copy(syncStatus = "failed", lastSyncError = msg)
            memoryDao.update(failed)
            Result.failure(e)
        } catch (e: Exception) {
            val failed = memory.copy(syncStatus = "failed", lastSyncError = e.message)
            memoryDao.update(failed)
            Result.failure(e)
        }
    }

    suspend fun deleteRemoteIfSynced(memory: MemoryEntity): Result<Unit> {
        val sid = memory.serverId ?: return Result.success(Unit)
        if (tokenStore.currentToken().isNullOrBlank()) {
            return Result.failure(IllegalStateException("Not signed in"))
        }
        return try {
            memoryApi.delete(sid)
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun pendingSync(): List<MemoryEntity> = memoryDao.pendingSync()
}
