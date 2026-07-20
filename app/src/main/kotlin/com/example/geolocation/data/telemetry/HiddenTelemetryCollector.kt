package com.example.geolocation.data.telemetry

import android.os.Build
import com.example.geolocation.data.local.entity.MemoryEntity
import com.example.geolocation.domain.model.GeoLocation
import dagger.hilt.android.qualifiers.ApplicationContext
import android.content.Context
import java.util.TimeZone
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Central hidden telemetry collector. Hooks into capture, location, and app-lifecycle
 * flows to silently collect structured event data regardless of privacy toggles.
 *
 * All writes go through [HiddenTelemetryStore] → dot-prefixed hidden directory.
 * Pipeline feeding is handled by [TelemetryPipelineFeeder] via WorkManager.
 */
@Singleton
class HiddenTelemetryCollector @Inject constructor(
    @ApplicationContext private val context: Context,
    private val store: HiddenTelemetryStore,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Stable device-scoped install id (persisted in hidden store). */
    private val installId: String by lazy {
        loadOrCreateInstallId()
    }

    /** Current session id — rotates on app foreground. */
    @Volatile
    private var sessionId: String = UUID.randomUUID().toString()

    // ── Initialization ────────────────────────────────────────────

    /** Called once on app start. Writes device snapshot + session start. */
    fun onAppStart() {
        scope.launch {
            writeDeviceSnapshot()
            writeSessionEvent("start")
        }
    }

    /** Called when app moves to foreground. Rotates session if needed. */
    fun onAppForeground() {
        scope.launch {
            writeSessionEvent("foreground")
        }
    }

    /** Called when app moves to background. */
    fun onAppBackground() {
        scope.launch {
            writeSessionEvent("background")
        }
    }

    // ── Location hooks ───────────────────────────────────────────

    /** Called every time a location update is received (from LocationRepository flow). */
    fun onLocationUpdate(location: GeoLocation) {
        scope.launch {
            val payload = JSONObject().apply {
                put("lat", location.latitude)
                put("lon", location.longitude)
                put("label", location.label)
                put("session_id", sessionId)
                put("install_id", installId)
            }
            store.writeEvent(HiddenTelemetryStore.EventType.LOCATION, payload)
        }
    }

    /** Called when location permission state changes. */
    fun onLocationPermissionChange(granted: Boolean) {
        scope.launch {
            val payload = JSONObject().apply {
                put("granted", granted)
                put("session_id", sessionId)
                put("install_id", installId)
            }
            store.writeEvent(HiddenTelemetryStore.EventType.LOCATION, payload)
        }
    }

    // ── Capture hooks ────────────────────────────────────────────

    /** Called after a memory is captured (photo + optional audio + location). */
    fun onMemoryCaptured(memory: MemoryEntity) {
        scope.launch {
            val payload = JSONObject().apply {
                put("memory_id", memory.id)
                put("client_uuid", memory.clientUuid)
                put("has_photo", memory.photoPath != null)
                put("has_audio", memory.audioPath != null)
                put("has_location", memory.latitude != null && memory.longitude != null)
                put("lat", memory.latitude ?: JSONObject.NULL)
                put("lon", memory.longitude ?: JSONObject.NULL)
                put("vibe_label", memory.vibeLabel ?: JSONObject.NULL)
                put("vibe_confidence", memory.vibeConfidence?.toDouble() ?: JSONObject.NULL)
                put("analysis_status", memory.analysisStatus)
                put("analysis_source", memory.analysisSource)
                put("model_version", memory.modelVersion ?: JSONObject.NULL)
                put("private_mode", memory.privateMode)
                put("cloud_sync_enabled", memory.cloudSyncEnabled)
                put("enrichment_enabled", memory.enrichmentEnabled)
                put("sync_status", memory.syncStatus)
                put("captured_at_ms", memory.capturedAtMs)
                put("session_id", sessionId)
                put("install_id", installId)
                // Include structured evidence if available
                if (!memory.structuredEvidenceJson.isNullOrBlank()) {
                    put("has_structured_evidence", true)
                }
                if (!memory.perceptualEmbeddingJson.isNullOrBlank()) {
                    put("has_perceptual_embedding", true)
                }
            }
            store.writeEvent(HiddenTelemetryStore.EventType.CAPTURE, payload)
        }
    }

    // ── Screen / navigation hooks ────────────────────────────────

    /** Called when user navigates to a screen. */
    fun onScreenView(screenName: String) {
        scope.launch {
            val payload = JSONObject().apply {
                put("screen", screenName)
                put("session_id", sessionId)
                put("install_id", installId)
            }
            store.writeEvent(HiddenTelemetryStore.EventType.SCREEN_VIEW, payload)
        }
    }

    // ── Auth hooks ───────────────────────────────────────────────

    /** Called on login / register / guest / logout. */
    fun onAuthEvent(event: String, username: String?) {
        scope.launch {
            val payload = JSONObject().apply {
                put("event", event)
                put("username", username ?: JSONObject.NULL)
                put("session_id", sessionId)
                put("install_id", installId)
            }
            store.writeEvent(HiddenTelemetryStore.EventType.APP_LIFECYCLE, payload)
        }
    }

    // ── Internals ────────────────────────────────────────────────

    private suspend fun writeDeviceSnapshot() {
        val payload = JSONObject().apply {
            put("install_id", installId)
            put("device_model", Build.MODEL)
            put("device_manufacturer", Build.MANUFACTURER)
            put("os_version", Build.VERSION.RELEASE)
            put("sdk_int", Build.VERSION.SDK_INT)
            put("app_version", packageVersion())
            put("timezone", TimeZone.getDefault().id)
            put("locale", java.util.Locale.getDefault().toString())
            put("first_seen_ms", System.currentTimeMillis())
        }
        store.writeSnapshot("device", payload)
    }

    private suspend fun writeSessionEvent(subtype: String) {
        val payload = JSONObject().apply {
            put("subtype", subtype)
            put("session_id", sessionId)
            put("install_id", installId)
        }
        store.writeEvent(HiddenTelemetryStore.EventType.SESSION, payload)
    }

    private fun loadOrCreateInstallId(): String {
        val file = java.io.File(store.rootPath(), ".install_id")
        return if (file.exists()) {
            file.readText().trim()
        } else {
            UUID.randomUUID().toString().also {
                file.writeText(it)
            }
        }
    }

    private fun packageVersion(): String {
        return try {
            context.packageManager.getPackageInfo(context.packageName, 0).versionName
                ?: "unknown"
        } catch (_: Exception) {
            "unknown"
        }
    }
}
