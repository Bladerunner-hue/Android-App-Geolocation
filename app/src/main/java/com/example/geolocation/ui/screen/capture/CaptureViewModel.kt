package com.example.geolocation.ui.screen.capture

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.repository.MemoryRepository
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

data class CaptureUiState(
    val photoPath: String? = null,
    val caption: String = "",
    val recordAudio: Boolean = false,
    val useLocation: Boolean = false,
    val saving: Boolean = false,
    val message: String? = null,
    val done: Boolean = false,
)

@HiltViewModel
class CaptureViewModel @Inject constructor(
    private val memoryRepository: MemoryRepository,
    @ApplicationContext private val context: Context,
) : ViewModel() {

    private val _uiState = MutableStateFlow(CaptureUiState())
    val uiState: StateFlow<CaptureUiState> = _uiState.asStateFlow()

    fun onCaptionChange(v: String) = _uiState.update { it.copy(caption = v) }
    fun onToggleAudio(v: Boolean) = _uiState.update { it.copy(recordAudio = v) }
    fun onUseLocation(v: Boolean) = _uiState.update { it.copy(useLocation = v) }
    fun onPhotoReady(file: File) = _uiState.update { it.copy(photoPath = file.absolutePath) }

    fun save() {
        viewModelScope.launch {
            _uiState.update { it.copy(saving = true, message = null) }
            try {
                val photo = _uiState.value.photoPath?.let { File(it) }
                val audioDir = File(context.filesDir, "captures").apply { mkdirs() }
                val audioFile = File(audioDir, "audio_${System.currentTimeMillis()}.wav")
                val loc = if (_uiState.value.useLocation) lastLocation() else null
                val mem = memoryRepository.capture(
                    photoFile = photo,
                    recordAudio = _uiState.value.recordAudio,
                    audioOut = audioFile,
                    latitude = loc?.latitude,
                    longitude = loc?.longitude,
                    caption = _uiState.value.caption.ifBlank { null },
                )
                _uiState.update {
                    it.copy(
                        saving = false,
                        done = true,
                        message = "Saved #${mem.id} · ${mem.analysisStatus} · sync=${mem.syncStatus}",
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(saving = false, message = e.message ?: "Save failed")
                }
            }
        }
    }

    @SuppressLint("MissingPermission")
    private suspend fun lastLocation(): Location? {
        return try {
            val client = LocationServices.getFusedLocationProviderClient(context)
            client.getCurrentLocation(
                Priority.PRIORITY_BALANCED_POWER_ACCURACY,
                CancellationTokenSource().token,
            ).await()
        } catch (_: Exception) {
            null
        }
    }
}
