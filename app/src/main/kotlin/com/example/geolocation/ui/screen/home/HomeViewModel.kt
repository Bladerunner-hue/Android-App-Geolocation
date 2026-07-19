package com.example.geolocation.ui.screen.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.repository.AuthRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class HomeUiState(
    val username: String = "Guest",
    val tokenPresent: Boolean = false,
)

@HiltViewModel
class HomeViewModel @Inject constructor(
    private val authRepository: AuthRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            combine(authRepository.token, authRepository.currentUsername) { token, name ->
                token to name
            }.collect { (token, name) ->
                _uiState.update {
                    it.copy(
                        username = when {
                            !name.isNullOrBlank() -> name
                            token.isNullOrBlank() -> "Guest"
                            else -> "User"
                        },
                        tokenPresent = !token.isNullOrBlank(),
                    )
                }
            }
        }
    }

    fun logout() {
        viewModelScope.launch {
            authRepository.logout()
        }
    }
}
