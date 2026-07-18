package com.example.geolocation.ui.screen.train

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.local.dao.MemoryTrainingLabelDao
import com.example.geolocation.data.local.entity.MemoryTrainingLabelEntity
import com.example.geolocation.data.ml.FusionV0Interpreter
import com.example.geolocation.domain.TrainModeSession
import dagger.hilt.android.lifecycle.HiltViewModel
import java.util.TimeZone
import java.util.UUID
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONArray

@HiltViewModel
class TrainModeViewModel @Inject constructor(
    private val labelDao: MemoryTrainingLabelDao,
    private val session: TrainModeSession,
    private val fusion: FusionV0Interpreter,
    savedStateHandle: SavedStateHandle,
) : ViewModel() {

    private val memoryId: String = savedStateHandle["memoryId"] ?: ""

    private val _uiState = MutableStateFlow(TrainModeUiState(memoryId = memoryId))
    val uiState: StateFlow<TrainModeUiState> = _uiState.asStateFlow()

    fun onSelectVibe(v: String) = _uiState.update { it.copy(primaryVibe = v) }
    fun onValence(v: Float) = _uiState.update { it.copy(valence = v) }
    fun onArousal(v: Float) = _uiState.update { it.copy(arousal = v) }
    fun onConfidence(v: Float) = _uiState.update { it.copy(confidence = v) }
    fun onConsentTraining(v: Boolean) = _uiState.update { it.copy(consentTraining = v) }
    fun onConsentCloud(v: Boolean) = _uiState.update { it.copy(consentCloud = v) }
    fun onNote(v: String) = _uiState.update { it.copy(note = v) }

    fun submitLabel() {
        val s = _uiState.value
        val vibe = s.primaryVibe ?: return
        viewModelScope.launch {
            val now = System.currentTimeMillis()
            val entity = MemoryTrainingLabelEntity(
                id = UUID.randomUUID().toString(),
                memoryId = s.memoryId.ifBlank { "unknown" },
                sessionId = session.current(now),
                primaryVibe = vibe,
                secondaryVibesJson = JSONArray().toString(),
                valence = s.valence.toInt(),
                arousal = s.arousal.toInt(),
                labelConfidence = s.confidence.toInt().coerceIn(1, 3),
                labelSource = "human_self",
                utcOffsetMinutes = TimeZone.getDefault().getOffset(now) / 60_000,
                locationAccuracyMeters = null,
                consentForTraining = s.consentTraining,
                consentForCloud = s.consentCloud,
                labelledAtEpochMillis = now,
            )
            labelDao.insert(entity)
            // Only now unlock model prediction (if any)
            val pred = if (fusion.isAvailable()) {
                "model packaged — run fusion after embeddings"
            } else {
                fusion.unavailableReason()
            }
            _uiState.update {
                it.copy(
                    predictionUnlocked = true,
                    modelPrediction = pred,
                    message = "Label saved. Thank you.",
                )
            }
        }
    }
}
