package com.example.geolocation.data.repository

import com.example.geolocation.data.local.PrivacyPreferences
import com.example.geolocation.data.local.dao.MemoryDao
import com.example.geolocation.data.local.entity.MemoryEntity
import com.example.geolocation.data.ml.AmbientAudioRecorder
import com.example.geolocation.data.ml.OnDeviceVibeAnalyzer
import com.example.geolocation.data.remote.api.MemoryApi
import com.example.geolocation.util.SyncScheduler
import java.io.File
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody

@Singleton
class MemoryRepository @Inject constructor(
    private val memoryDao: MemoryDao,
    private val privacy: PrivacyPreferences,
    private val analyzer: OnDeviceVibeAnalyzer,
    private val audioRecorder: AmbientAudioRecorder,
    private val memoryApi: MemoryApi,
    private val syncScheduler: SyncScheduler,
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
    ): MemoryEntity {
        val snap = privacy.snapshot.first()
        var audioPath: String? = null
        if (recordAudio && snap.audioCaptureEnabled && audioOut != null) {
            val result = audioRecorder.recordToFile(audioOut)
            audioPath = result?.file?.absolutePath
        }
        val analysis = analyzer.analyzeMedia(
            photo = photoFile,
            audio = audioPath?.let { File(it) },
            hasLocation = latitude != null && longitude != null,
        )
        val (vibe, conf, status) = when (analysis) {
            is OnDeviceVibeAnalyzer.AnalysisResult.Available ->
                Triple(analysis.vibeLabel, analysis.confidence, "on_device")
            is OnDeviceVibeAnalyzer.AnalysisResult.Unavailable ->
                Triple(null, null, "unavailable")
        }
        val now = System.currentTimeMillis()
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
            analysisStatus = status,
            privateMode = snap.privateMode,
            cloudSyncEnabled = snap.cloudSyncEnabled && !snap.privateMode,
            enrichmentEnabled = snap.enrichmentEnabled && snap.cloudSyncEnabled && !snap.privateMode,
            capturedAtMs = now,
            createdAtMs = now,
            syncStatus = syncStatus,
            lastSyncError = if (analysis is OnDeviceVibeAnalyzer.AnalysisResult.Unavailable) {
                analysis.reason
            } else {
                null
            },
            serverId = null,
            evidenceJson = null,
        )
        val id = memoryDao.insert(entity)
        val saved = entity.copy(id = id)
        if (saved.syncStatus == "pending") {
            syncScheduler.enqueueMemorySync()
        }
        return saved
    }

    suspend fun syncOne(memory: MemoryEntity): Result<MemoryEntity> {
        if (memory.privateMode || !memory.cloudSyncEnabled) {
            return Result.failure(IllegalStateException("Private or sync disabled"))
        }
        return try {
            val text = { s: String -> s.toRequestBody("text/plain".toMediaTypeOrNull()) }
            val photoPart = memory.photoPath?.let { path ->
                val f = File(path)
                if (!f.exists()) return@let null
                MultipartBody.Part.createFormData(
                    "photo",
                    f.name,
                    f.asRequestBody("image/*".toMediaTypeOrNull()),
                )
            }
            val audioPart = memory.audioPath?.let { path ->
                val f = File(path)
                if (!f.exists()) return@let null
                MultipartBody.Part.createFormData(
                    "audio",
                    f.name,
                    f.asRequestBody("audio/wav".toMediaTypeOrNull()),
                )
            }
            val dto = memoryApi.analyze(
                clientUuid = text(memory.clientUuid),
                privateMode = text("false"),
                caption = memory.caption?.let { text(it) },
                latitude = memory.latitude?.toString()?.let { text(it) },
                longitude = memory.longitude?.toString()?.let { text(it) },
                vibe = memory.vibeLabel?.let { text(it) },
                confidence = memory.vibeConfidence?.toString()?.let { text(it) },
                enrichment = text(memory.enrichmentEnabled.toString()),
                photo = photoPart,
                audio = audioPart,
            )
            val updated = memory.copy(
                serverId = dto.id,
                syncStatus = "synced",
                lastSyncError = null,
                analysisStatus = dto.analysisStatus,
                vibeLabel = dto.vibeLabel ?: memory.vibeLabel,
            )
            memoryDao.update(updated)
            Result.success(updated)
        } catch (e: Exception) {
            val failed = memory.copy(syncStatus = "failed", lastSyncError = e.message)
            memoryDao.update(failed)
            Result.failure(e)
        }
    }

    suspend fun pendingSync(): List<MemoryEntity> = memoryDao.pendingSync()
}
