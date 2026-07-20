package com.example.geolocation.ui.screen.capture

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import androidx.core.content.ContextCompat
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
    /** Explicit location permission for this workflow (not silent). */
    val locationPermissionGranted: Boolean = false,
    val locationStatus: String = "off",
    val saving: Boolean = false,
    val message: String? = null,
    val done: Boolean = false,
)

@HiltViewModel
class CaptureViewModel @Inject constructor(
    private val memoryRepository: MemoryRepository,
    @ApplicationContext private val context: Context,
) : ViewModel() {

    private val _uiState = MutableStateFlow(
        CaptureUiState(locationPermissionGranted = hasLocationPermission()),
    )
    val uiState: StateFlow<CaptureUiState> = _uiState.asStateFlow()

    fun onCaptionChange(v: String) = _uiState.update { it.copy(caption = v) }
    fun onToggleAudio(v: Boolean) = _uiState.update { it.copy(recordAudio = v) }
    fun onPhotoReady(file: File) = _uiState.update { it.copy(photoPath = file.absolutePath) }

    fun onUseLocation(v: Boolean) {
        _uiState.update {
            it.copy(
                useLocation = v && it.locationPermissionGranted,
                locationStatus = when {
                    !v -> "off"
                    !it.locationPermissionGranted -> "permission_required"
                    else -> "requested"
                },
                message = if (v && !it.locationPermissionGranted) {
                    "Location permission required to attach GPS"
                } else {
                    it.message
                },
            )
        }
    }

    fun onLocationPermissionResult(granted: Boolean) {
        _uiState.update {
            it.copy(
                locationPermissionGranted = granted,
                useLocation = if (granted) it.useLocation else false,
                locationStatus = when {
                    granted && it.useLocation -> "granted"
                    granted -> "ready"
                    else -> "denied"
                },
                message = if (!granted) "Location permission denied — saving without GPS" else it.message,
            )
        }
    }

    fun save() {
        viewModelScope.launch {
            _uiState.update { it.copy(saving = true, message = null) }
            try {
                val photo = _uiState.value.photoPath?.let { File(it) }
                val audioDir = File(context.filesDir, "captures").apply { mkdirs() }
                val audioFile = File(audioDir, "audio_${System.currentTimeMillis()}.wav")
                val wantLoc = _uiState.value.useLocation && _uiState.value.locationPermissionGranted
                val loc = if (wantLoc) lastLocation() else null
                if (wantLoc && loc == null) {
                    _uiState.update { it.copy(locationStatus = "unavailable") }
                }
                val mem = memoryRepository.capture(
                    photoFile = photo,
                    recordAudio = _uiState.value.recordAudio,
                    audioOut = audioFile,
                    latitude = loc?.latitude,
                    longitude = loc?.longitude,
                    caption = _uiState.value.caption.ifBlank { null },
                    locationAccuracyM = loc?.accuracy?.takeIf { it > 0f },
                )
                _uiState.update {
                    it.copy(
                        saving = false,
                        done = true,
                        message = "Saved #${mem.id} · ${mem.analysisStatus} · sync=${mem.syncStatus}" +
                            (loc?.let { l -> " · ±${l.accuracy.toInt()}m" } ?: ""),
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(saving = false, message = e.message ?: "Save failed")
                }
            }
        }
    }

    private fun hasLocationPermission(): Boolean {
        val fine = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION,
        ) == PackageManager.PERMISSION_GRANTED
        val coarse = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_COARSE_LOCATION,
        ) == PackageManager.PERMISSION_GRANTED
        return fine || coarse
    }

    @SuppressLint("MissingPermission")
    private suspend fun lastLocation(): Location? {
        if (!hasLocationPermission()) return null
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
