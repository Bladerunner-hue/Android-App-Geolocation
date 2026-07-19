package com.example.geolocation.ui.screen.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geolocation.data.repository.AuthRepository
import com.example.geolocation.domain.usecase.LoginUseCase
import com.example.geolocation.util.Result
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class LoginUiState(
    val username: String = "",
    val password: String = "",
    val email: String = "",
    val isRegisterMode: Boolean = false,
    val isLoading: Boolean = false,
    val isAuthenticated: Boolean = false,
    /** Offline free path — no JWT. */
    val continueAsGuest: Boolean = false,
    val error: String? = null,
    val token: String? = null,
)

@HiltViewModel
class LoginViewModel @Inject constructor(
    private val loginUseCase: LoginUseCase,
    private val authRepository: AuthRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(LoginUiState())
    val uiState: StateFlow<LoginUiState> = _uiState.asStateFlow()

    fun onUsernameChanged(value: String) {
        _uiState.update { it.copy(username = value, error = null) }
    }

    fun onPasswordChanged(value: String) {
        _uiState.update { it.copy(password = value, error = null) }
    }

    fun onEmailChanged(value: String) {
        _uiState.update { it.copy(email = value, error = null) }
    }

    fun toggleRegisterMode() {
        _uiState.update {
            it.copy(isRegisterMode = !it.isRegisterMode, error = null)
        }
    }

    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }

    fun continueOffline() {
        viewModelScope.launch {
            authRepository.continueOffline()
            _uiState.update {
                it.copy(continueAsGuest = true, isAuthenticated = false, error = null)
            }
        }
    }

    fun login() {
        val username = _uiState.value.username
        val password = _uiState.value.password

        viewModelScope.launch {
            loginUseCase(username, password).collect { result ->
                when (result) {
                    is Result.Loading -> _uiState.update { it.copy(isLoading = true, error = null) }
                    is Result.Success -> _uiState.update {
                        it.copy(
                            isLoading = false,
                            isAuthenticated = true,
                            continueAsGuest = false,
                            token = result.data.token,
                            error = null,
                        )
                    }
                    is Result.Error -> _uiState.update {
                        it.copy(isLoading = false, error = result.message)
                    }
                }
            }
        }
    }

    fun register() {
        val s = _uiState.value
        if (s.username.isBlank() || s.password.isBlank() || s.email.isBlank()) {
            _uiState.update { it.copy(error = "Username, email, and password are required") }
            return
        }
        viewModelScope.launch {
            authRepository.register(s.username.trim(), s.email.trim(), s.password).collect { result ->
                when (result) {
                    is Result.Loading -> _uiState.update { it.copy(isLoading = true, error = null) }
                    is Result.Success -> _uiState.update {
                        it.copy(
                            isLoading = false,
                            isAuthenticated = true,
                            continueAsGuest = false,
                            token = result.data.token,
                            error = null,
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
