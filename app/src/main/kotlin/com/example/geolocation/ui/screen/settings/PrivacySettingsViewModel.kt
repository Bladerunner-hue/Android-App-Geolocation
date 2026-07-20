package com.example.geolocation.ui.screen.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.local.PrivacyPreferences
import com.example.geolocation.data.telemetry.TrainingBronzeExporter
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class PrivacyUiState(
    val privateMode: Boolean = true,
    val cloudSync: Boolean = false,
    val enrichment: Boolean = false,
    val audioCapture: Boolean = false,
    val exportBusy: Boolean = false,
    val exportMessage: String? = null,
)

@HiltViewModel
class PrivacySettingsViewModel @Inject constructor(
    private val privacy: PrivacyPreferences,
    private val bronzeExporter: TrainingBronzeExporter,
) : ViewModel() {

    private val exportBusy = MutableStateFlow(false)
    private val exportMessage = MutableStateFlow<String?>(null)

    val uiState: StateFlow<PrivacyUiState> = combine(
        privacy.snapshot,
        exportBusy,
        exportMessage,
    ) { snap, busy, msg ->
        PrivacyUiState(
            privateMode = snap.privateMode,
            cloudSync = snap.cloudSyncEnabled,
            enrichment = snap.enrichmentEnabled,
            audioCapture = snap.audioCaptureEnabled,
            exportBusy = busy,
            exportMessage = msg,
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), PrivacyUiState())

    fun setPrivateMode(v: Boolean) = viewModelScope.launch { privacy.setPrivateMode(v) }
    fun setCloudSync(v: Boolean) = viewModelScope.launch { privacy.setCloudSyncEnabled(v) }
    fun setEnrichment(v: Boolean) = viewModelScope.launch { privacy.setEnrichmentEnabled(v) }
    fun setAudio(v: Boolean) = viewModelScope.launch { privacy.setAudioCaptureEnabled(v) }

    fun clearExportMessage() = exportMessage.update { null }

    /** Local bronze export for Train Mode labels with consent_for_training. */
    fun exportTrainingBronze() {
        viewModelScope.launch {
            exportBusy.value = true
            exportMessage.value = null
            try {
                val result = bronzeExporter.exportConsentedBronze(
                    includeMedia = true,
                    createZip = true,
                )
                val zipPath = result.zipFile?.absolutePath ?: result.exportDir.absolutePath
                exportMessage.value =
                    "Exported ${result.rowCount} rows · ${result.classCounts} → $zipPath"
            } catch (e: Exception) {
                exportMessage.value = e.message ?: "Export failed"
            } finally {
                exportBusy.value = false
            }
        }
    }
}
