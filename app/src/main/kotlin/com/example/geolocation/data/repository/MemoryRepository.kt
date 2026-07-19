package com.example.geolocation.data.repository

import com.example.geolocation.data.local.PrivacyPreferences
import com.example.geolocation.data.local.TokenStore
import com.example.geolocation.data.local.dao.MemoryDao
import com.example.geolocation.data.local.entity.MemoryEntity
import com.example.geolocation.data.ml.AmbientAudioRecorder
import com.example.geolocation.data.ml.ContextEncoderV1
import com.example.geolocation.data.ml.FusionV0Interpreter
import com.example.geolocation.data.ml.OnDeviceVibeAnalyzer
import com.example.geolocation.data.remote.api.MemoryApi
import com.example.geolocation.data.remote.mapper.MemoryAnalyzeMapper
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
    private val analyzer: OnDeviceVibeAnalyzer,
    private val fusion: FusionV0Interpreter,
    private val audioRecorder: AmbientAudioRecorder,
    private val memoryApi: MemoryApi,
    private val syncScheduler: SyncScheduler,
    private val tokenStore: TokenStore,
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
            accuracyM = locationAccuracyM,
        )
        val mask = floatArrayOf(
            if (photoFile != null) 1f else 0f,
            if (audioPath != null) 1f else 0f,
            if (latitude != null && longitude != null) 1f else 0f,
        )

        // Prefer FusionV0Interpreter contract; fall back to analyzer media path.
        var vibe: String? = null
        var conf: Float? = null
        var probs: FloatArray? = null
        var perceptual: FloatArray? = null
        var modelVersion: String? = null
        var analysisSource = "unavailable"
        var analysisStatus = "unavailable"
        var lastError: String? = null

        when (val fr = tryFusionOrAnalyzer(photoFile, audioPath, latitude != null)) {
            is EdgeAnalysis.Available -> {
                vibe = fr.vibeLabel
                conf = fr.confidence
                probs = fr.probs
                perceptual = fr.perceptual
                modelVersion = fr.modelVersion
                analysisSource = "on_device"
                analysisStatus = "on_device"
            }
            is EdgeAnalysis.Unavailable -> {
                lastError = fr.reason
            }
        }

        val structured = MemoryAnalyzeMapper.buildStructuredEvidence(
            vibeProbs = probs,
            context12 = context12,
            modalityMask = mask,
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
            structuredEvidenceJson = structured,
        )
        val id = memoryDao.insert(entity)
        val saved = entity.copy(id = id)
        if (saved.syncStatus == "pending") {
            syncScheduler.enqueueMemorySync()
        }
        return saved
    }

    private sealed class EdgeAnalysis {
        data class Available(
            val vibeLabel: String,
            val confidence: Float,
            val probs: FloatArray?,
            val perceptual: FloatArray?,
            val modelVersion: String?,
        ) : EdgeAnalysis()

        data class Unavailable(val reason: String) : EdgeAnalysis()
    }

    private fun tryFusionOrAnalyzer(
        photo: File?,
        audioPath: String?,
        hasLocation: Boolean,
    ): EdgeAnalysis {
        // Full fusion needs precomputed embeddings; without extractors, stay honest.
        if (fusion.isAvailable()) {
            // Interpreter present but raw media path still needs MobileNet/YAMNet — unavailable.
            return EdgeAnalysis.Unavailable(
                fusion.unavailableReason()
                    ?: "fusion_v0 packaged; edge embedding extractors not bundled",
            )
        }
        return when (
            val a = analyzer.analyzeMedia(
                photo = photo,
                audio = audioPath?.let { File(it) },
                hasLocation = hasLocation,
            )
        ) {
            is OnDeviceVibeAnalyzer.AnalysisResult.Available ->
                EdgeAnalysis.Available(
                    vibeLabel = a.vibeLabel,
                    confidence = a.confidence,
                    probs = a.probs,
                    perceptual = null,
                    modelVersion = "fusion_v0",
                )
            is OnDeviceVibeAnalyzer.AnalysisResult.Unavailable ->
                EdgeAnalysis.Unavailable(a.reason)
        }
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

    suspend fun pendingSync(): List<MemoryEntity> = memoryDao.pendingSync()
}
