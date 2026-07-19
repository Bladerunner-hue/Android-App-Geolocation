package com.example.geolocation.ui.screen.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.repository.LocationRepository
import com.example.geolocation.domain.model.GeoLocation
import com.example.geolocation.util.Result
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class DashboardUiState(
    val quickActions: List<QuickAction> = listOf(
        QuickAction("Location", "Start tracking"),
        QuickAction("Camera", "Capture a moment"),
        QuickAction("History", "View logs"),
        QuickAction("Settings", "Adjust preferences")
    ),
    val recentLocations: List<GeoLocation> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null
)

data class QuickAction(
    val title: String,
    val description: String
)

@HiltViewModel
class DashboardViewModel @Inject constructor(
    private val locationRepository: LocationRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState.asStateFlow()

    init {
        observeLocations()
    }

    private fun observeLocations() {
        viewModelScope.launch {
            locationRepository.observeLocations().collect { result ->
                when (result) {
                    is Result.Loading -> _uiState.update { it.copy(isLoading = true, error = null) }
                    is Result.Success -> _uiState.update {
                        it.copy(
                            isLoading = false,
                            error = null,
                            recentLocations = (it.recentLocations + result.data).takeLast(5)
                        )
                    }
                    is Result.Error -> _uiState.update {
                        it.copy(isLoading = false, error = result.message)
                    }
                }
            }
        }
    }
}
