package com.example.geolocation.ui.screen.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.local.PrivacyPreferences
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

@HiltViewModel
class PrivacySettingsViewModel @Inject constructor(
    private val privacy: PrivacyPreferences,
) : ViewModel() {

    val uiState: StateFlow<PrivacyUiState> = privacy.snapshot
        .map {
            PrivacyUiState(
                privateMode = it.privateMode,
                cloudSync = it.cloudSyncEnabled,
                enrichment = it.enrichmentEnabled,
                audioCapture = it.audioCaptureEnabled,
            )
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), PrivacyUiState())

    fun setPrivateMode(v: Boolean) = viewModelScope.launch { privacy.setPrivateMode(v) }
    fun setCloudSync(v: Boolean) = viewModelScope.launch { privacy.setCloudSyncEnabled(v) }
    fun setEnrichment(v: Boolean) = viewModelScope.launch { privacy.setEnrichmentEnabled(v) }
    fun setAudio(v: Boolean) = viewModelScope.launch { privacy.setAudioCaptureEnabled(v) }
}
