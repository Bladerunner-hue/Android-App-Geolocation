package com.example.geolocation.ui.screen.location

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.repository.LocationRepository
import com.example.geolocation.domain.model.GeoLocation
import com.example.geolocation.domain.usecase.GetLocationUseCase
import com.example.geolocation.util.Result
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class LocationUiState(
    val currentLocation: GeoLocation? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val hasPermission: Boolean = false
)

@HiltViewModel
class LocationViewModel @Inject constructor(
    private val getLocationUseCase: GetLocationUseCase,
    private val locationRepository: LocationRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(LocationUiState())
    val uiState: StateFlow<LocationUiState> = _uiState.asStateFlow()

    init {
        checkPermissionAndStart()
    }

    fun checkPermissionAndStart() {
        val hasPermission = locationRepository.hasLocationPermission()
        _uiState.update { it.copy(hasPermission = hasPermission) }
        if (hasPermission) {
            observe()
        }
    }

    fun onPermissionGranted() {
        _uiState.update { it.copy(hasPermission = true) }
        observe()
    }

    fun refreshLocation() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true) }
            val refreshed = getLocationUseCase.refresh()
            _uiState.update { it.copy(isLoading = false, currentLocation = refreshed) }
        }
    }

    private fun observe() {
        viewModelScope.launch {
            getLocationUseCase().collect { result ->
                when (result) {
                    is Result.Loading -> _uiState.update { it.copy(isLoading = true, error = null) }
                    is Result.Success -> _uiState.update {
                        it.copy(isLoading = false, currentLocation = result.data, error = null)
                    }
                    is Result.Error -> _uiState.update {
                        it.copy(isLoading = false, error = result.message)
                    }
                }
            }
        }
    }
}
