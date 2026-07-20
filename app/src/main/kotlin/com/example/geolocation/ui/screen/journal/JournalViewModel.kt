package com.example.geolocation.ui.screen.journal

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.local.PrivacyPreferences
import com.example.geolocation.data.local.entity.MemoryEntity
import com.example.geolocation.data.ml.EdgeMemoryAnalyzer
import com.example.geolocation.data.repository.MemoryRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class JournalUiState(
    val memories: List<MemoryEntity> = emptyList(),
    val query: String = "",
    val privateMode: Boolean = true,
    val mlStatus: String = "checking…",
)

@HiltViewModel
class JournalViewModel @Inject constructor(
    private val memoryRepository: MemoryRepository,
    private val privacy: PrivacyPreferences,
    private val edgeAnalyzer: EdgeMemoryAnalyzer,
) : ViewModel() {

    private val _uiState = MutableStateFlow(JournalUiState())
    val uiState: StateFlow<JournalUiState> = _uiState.asStateFlow()

    init {
        val cap = edgeAnalyzer.capability()
        val ml = when {
            cap.modelReady && cap.extractorsReady -> "edge ready (fusion_v0 + encoders)"
            cap.modelReady -> "fusion_v0 packaged; encoders missing"
            else -> "unavailable"
        }
        _uiState.update { it.copy(mlStatus = ml) }

        viewModelScope.launch {
            combine(
                memoryRepository.observeMemories(),
                privacy.privateMode,
            ) { memories, private ->
                memories to private
            }.collect { (memories, private) ->
                _uiState.update {
                    it.copy(
                        memories = if (it.query.isBlank()) memories else it.memories,
                        privateMode = private,
                    )
                }
                if (_uiState.value.query.isBlank()) {
                    _uiState.update { it.copy(memories = memories) }
                }
            }
        }
    }

    fun onQueryChange(q: String) {
        _uiState.update { it.copy(query = q) }
        viewModelScope.launch {
            val results = if (q.isBlank()) {
                memoryRepository.searchLocal("")
            } else {
                memoryRepository.searchLocal(q)
            }
            _uiState.update { it.copy(memories = results) }
        }
    }

    fun search() = onQueryChange(_uiState.value.query)
}
