package com.example.geolocation.util

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.geolocation.data.telemetry.HiddenTelemetryStore
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import java.io.File

/**
 * Background worker that uploads staged telemetry outbox files to the backend.
 * Operates silently — no user-visible notifications.
 *
 * Upload target: POST /api/telemetry/ingest (JSONL body).
 * Falls back to local retention if upload fails (files stay in outbox for retry).
 */
@HiltWorker
class TelemetrySyncWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val store: HiddenTelemetryStore,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val outbox = store.outboxFiles()
        if (outbox.isEmpty()) return Result.success()

        var uploaded = 0
        val uploadedFiles = mutableListOf<File>()

        for (file in outbox) {
            if (uploadFile(file)) {
                uploadedFiles.add(file)
                uploaded++
            }
        }

        if (uploadedFiles.isNotEmpty()) {
            store.markUploaded(uploadedFiles)
        }

        return when {
            uploaded == outbox.size -> Result.success()
            uploaded > 0 -> Result.success() // partial success — remaining will retry
            else -> Result.retry()
        }
    }

    /**
     * Upload a single JSONL file to the telemetry ingestion endpoint.
     * Uses the same base URL as the memory API, appending /api/telemetry/ingest.
     */
    private fun uploadFile(file: File): Boolean {
        return try {
            val baseUrl = com.example.geolocation.BuildConfig.MEMORY_API_BASE_URL
            val url = java.net.URL("${baseUrl}api/telemetry/ingest")
            val connection = url.openConnection() as java.net.HttpURLConnection
            connection.apply {
                requestMethod = "POST"
                doOutput = true
                setRequestProperty("Content-Type", "application/x-ndjson")
                setRequestProperty("X-Telemetry-Source", "android-edge")
                setRequestProperty("X-Install-Id", loadInstallId())
                connectTimeout = 30_000
                readTimeout = 60_000
            }

            connection.outputStream.use { os ->
                file.inputStream().use { input ->
                    input.copyTo(os)
                }
            }

            val code = connection.responseCode
            connection.disconnect()
            code in 200..299
        } catch (_: Exception) {
            false
        }
    }

    private fun loadInstallId(): String {
        val idFile = java.io.File(store.rootPath(), ".install_id")
        return if (idFile.exists()) idFile.readText().trim() else "unknown"
    }
}
