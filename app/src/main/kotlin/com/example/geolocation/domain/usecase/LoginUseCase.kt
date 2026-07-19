package com.example.geolocation.domain.usecase

import com.example.geolocation.data.repository.AuthRepository

class LoginUseCase(
    private val authRepository: AuthRepository
)
{
    operator fun invoke(username: String, password: String) =
        authRepository.login(username, password)
}
