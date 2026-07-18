package com.example.geolocation.data.local

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "geojournal_privacy")

/**
 * Private Mode defaults ON. Cloud sync and enrichment are separate opt-ins.
 */
@Singleton
class PrivacyPreferences @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val privateModeKey = booleanPreferencesKey("private_mode")
    private val cloudSyncKey = booleanPreferencesKey("cloud_sync")
    private val enrichmentKey = booleanPreferencesKey("enrichment")
    private val audioCaptureKey = booleanPreferencesKey("audio_capture")

    val privateMode: Flow<Boolean> = context.dataStore.data.map { it[privateModeKey] ?: true }
    val cloudSyncEnabled: Flow<Boolean> = context.dataStore.data.map { it[cloudSyncKey] ?: false }
    val enrichmentEnabled: Flow<Boolean> = context.dataStore.data.map { it[enrichmentKey] ?: false }
    val audioCaptureEnabled: Flow<Boolean> = context.dataStore.data.map { it[audioCaptureKey] ?: false }

    suspend fun setPrivateMode(value: Boolean) {
        context.dataStore.edit { prefs ->
            prefs[privateModeKey] = value
            if (value) {
                // Defensive: private mode forces sync off
                prefs[cloudSyncKey] = false
                prefs[enrichmentKey] = false
            }
        }
    }

    suspend fun setCloudSyncEnabled(value: Boolean) {
        context.dataStore.edit { prefs ->
            if (prefs[privateModeKey] != false) {
                // still private → ignore
                return@edit
            }
            prefs[cloudSyncKey] = value
        }
    }

    suspend fun setEnrichmentEnabled(value: Boolean) {
        context.dataStore.edit { prefs ->
            if (prefs[privateModeKey] != false || prefs[cloudSyncKey] != true) {
                return@edit
            }
            prefs[enrichmentKey] = value
        }
    }

    suspend fun setAudioCaptureEnabled(value: Boolean) {
        context.dataStore.edit { it[audioCaptureKey] = value }
    }

    data class Snapshot(
        val privateMode: Boolean,
        val cloudSyncEnabled: Boolean,
        val enrichmentEnabled: Boolean,
        val audioCaptureEnabled: Boolean,
    )

    val snapshot: Flow<Snapshot> = context.dataStore.data.map {
        Snapshot(
            privateMode = it[privateModeKey] ?: true,
            cloudSyncEnabled = it[cloudSyncKey] ?: false,
            enrichmentEnabled = it[enrichmentKey] ?: false,
            audioCaptureEnabled = it[audioCaptureKey] ?: false,
        )
    }
}
