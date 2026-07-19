package com.example.geolocation.ui.screen.train

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.ml.FusionV0Interpreter
import com.example.geolocation.data.repository.TrainingLabelRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

@HiltViewModel
class TrainModeViewModel @Inject constructor(
    private val trainingLabelRepository: TrainingLabelRepository,
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
            try {
                trainingLabelRepository.submitLabel(
                    memoryId = s.memoryId.ifBlank { "unknown" },
                    primaryVibe = vibe,
                    valence = s.valence.toInt(),
                    arousal = s.arousal.toInt(),
                    confidence = s.confidence.toInt(),
                    consentTraining = s.consentTraining,
                    consentCloud = s.consentCloud,
                    noteIgnored = s.note,
                )
                val pred = if (fusion.isAvailable()) {
                    "model packaged — run fusion after embeddings"
                } else {
                    fusion.unavailableReason() ?: "model unavailable"
                }
                _uiState.update {
                    it.copy(
                        predictionUnlocked = true,
                        modelPrediction = pred,
                        message = "Label saved. Thank you.",
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(message = e.message ?: "Save failed") }
            }
        }
    }
}
