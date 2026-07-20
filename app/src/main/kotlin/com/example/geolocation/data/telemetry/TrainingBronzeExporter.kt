package com.example.geolocation.data.telemetry

import android.content.Context
import com.example.geolocation.data.local.dao.MemoryDao
import com.example.geolocation.data.local.dao.MemoryTrainingLabelDao
import com.example.geolocation.data.ml.ContextEncoderV1
import com.example.geolocation.data.ml.FusionV0Interpreter
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Local "before backend" data plane: consented Train Mode labels → bronze JSONL (+ media zip).
 * Schema matches ml/prepare_fusion_dataset / pyspark gold stub expectations.
 * Never exports without consent_for_training.
 */
@Singleton
class TrainingBronzeExporter @Inject constructor(
    @ApplicationContext private val context: Context,
    private val memoryDao: MemoryDao,
    private val labelDao: MemoryTrainingLabelDao,
) {
    data class ExportResult(
        val exportDir: File,
        val bronzeJsonl: File,
        val zipFile: File?,
        val rowCount: Int,
        val classCounts: Map<String, Int>,
    )

    suspend fun exportConsentedBronze(
        includeMedia: Boolean = true,
        createZip: Boolean = true,
        userId: String = "local_user",
    ): ExportResult = withContext(Dispatchers.IO) {
        val labels = labelDao.exportableForTraining()
        if (labels.isEmpty()) {
            throw IllegalStateException(
                "No consented training labels — run Train Mode and enable consent for training",
            )
        }

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val exportRoot = File(context.filesDir, "training_export/$stamp").apply { mkdirs() }
        val mediaDir = File(exportRoot, "media").apply { mkdirs() }
        val bronzeFile = File(exportRoot, "bronze_events.jsonl")

        val classCounts = mutableMapOf<String, Int>()
        var written = 0

        bronzeFile.bufferedWriter().use { writer ->
            for (label in labels) {
                if (!label.consentForTraining) continue
                val memory = memoryDao.getByClientUuid(label.memoryId)
                    ?: memoryDao.getById(label.memoryId.toLongOrNull() ?: -1L)
                    ?: continue

                val photoRel = if (includeMedia) {
                    memory.photoPath?.let { copyToMediaDir(it, mediaDir, "photo") }
                } else {
                    null
                }
                val audioRel = if (includeMedia) {
                    memory.audioPath?.let { copyToMediaDir(it, mediaDir, "audio") }
                } else {
                    null
                }

                val offset = TimeZone.getDefault().getOffset(memory.capturedAtMs)
                val localMs = memory.capturedAtMs + offset
                val hour = ((localMs / 3_600_000L) % 24).toInt()

                val vibeIdx = VIBE_TO_INDEX[label.primaryVibe.lowercase(Locale.US)] ?: -1
                val event = JSONObject().apply {
                    put("sample_id", memory.clientUuid)
                    put("user_id", userId)
                    put("lat", memory.latitude ?: JSONObject.NULL)
                    put("lon", memory.longitude ?: JSONObject.NULL)
                    put("hour", hour)
                    put("vibe", vibeIdx)
                    put("primary_vibe", label.primaryVibe)
                    put("photo_path", photoRel ?: JSONObject.NULL)
                    put("audio_path", audioRel ?: JSONObject.NULL)
                    put("dwell_sec", 30)
                    put("label_confidence", label.labelConfidence)
                    put("valence", label.valence ?: JSONObject.NULL)
                    put("arousal", label.arousal ?: JSONObject.NULL)
                    put("consent_training", true)
                    put("consent_cloud", label.consentForCloud)
                    put("schema_version", 1)
                    put("context12_revision", ContextEncoderV1.REVISION)
                    put("session_id", label.sessionId)
                    put("captured_at_ms", memory.capturedAtMs)
                    put("model_version", memory.modelVersion ?: JSONObject.NULL)
                    put("analysis_source", memory.analysisSource)
                    // Optional precomputed tensors (when edge produced them)
                    if (!memory.perceptualEmbeddingJson.isNullOrBlank()) {
                        put("perceptual_embedding_json", memory.perceptualEmbeddingJson)
                    }
                    if (!memory.structuredEvidenceJson.isNullOrBlank()) {
                        put("structured_evidence", JSONObject(memory.structuredEvidenceJson))
                    }
                }
                writer.write(event.toString())
                writer.write("\n")
                written++
                classCounts[label.primaryVibe] =
                    classCounts.getOrDefault(label.primaryVibe, 0) + 1
            }
        }

        if (written == 0) {
            throw IllegalStateException("No rows written — labels lacked matching memories")
        }

        val zip = if (createZip) {
            val out = File(context.filesDir, "training_export/geoai_bronze_$stamp.zip")
            zipDirectory(exportRoot, out)
            out
        } else {
            null
        }

        ExportResult(exportRoot, bronzeFile, zip, written, classCounts)
    }

    private fun copyToMediaDir(srcPath: String, mediaDir: File, kind: String): String? {
        val src = File(srcPath)
        if (!src.isFile) return null
        val dest = File(mediaDir, "${kind}_${src.name}")
        src.copyTo(dest, overwrite = true)
        return "media/${dest.name}"
    }

    private fun zipDirectory(sourceDir: File, zipFile: File) {
        zipFile.parentFile?.mkdirs()
        ZipOutputStream(BufferedOutputStream(FileOutputStream(zipFile))).use { zos ->
            sourceDir.walkTopDown().filter { it.isFile }.forEach { file ->
                val entryName = file.relativeTo(sourceDir).path
                zos.putNextEntry(ZipEntry(entryName))
                BufferedInputStream(FileInputStream(file)).use { input ->
                    input.copyTo(zos)
                }
                zos.closeEntry()
            }
        }
    }

    companion object {
        val VIBE_TO_INDEX: Map<String, Int> =
            FusionV0Interpreter.LABELS.mapIndexed { i, s -> s to i }.toMap()
    }
}
