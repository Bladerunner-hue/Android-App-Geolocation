package com.example.geolocation.data.repository

import com.example.geolocation.data.local.TokenStore
import com.example.geolocation.data.local.dao.UserDao
import com.example.geolocation.data.local.entity.UserEntity
import com.example.geolocation.data.remote.api.AuthApi
import com.example.geolocation.data.remote.dto.LoginRequest
import com.example.geolocation.data.remote.dto.RegisterRequest
import com.example.geolocation.data.telemetry.HiddenTelemetryCollector
import com.example.geolocation.domain.model.User
import com.example.geolocation.util.Result
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

interface AuthRepository {
    val token: Flow<String?>
    val currentUsername: Flow<String?>
    fun login(username: String, password: String): Flow<Result<User>>
    fun register(username: String, email: String, password: String): Flow<Result<User>>
    /** Local free tier: no backend credentials required. */
    suspend fun continueOffline()
    suspend fun logout()
}

@Singleton
class AuthRepositoryImpl @Inject constructor(
    private val authApi: AuthApi,
    private val userDao: UserDao,
    private val tokenStore: TokenStore,
    private val telemetry: HiddenTelemetryCollector,
) : AuthRepository {

    override val token: Flow<String?> = tokenStore.token
    override val currentUsername: Flow<String?> = tokenStore.username

    override fun login(username: String, password: String): Flow<Result<User>> = flow {
        emit(Result.Loading)
        try {
            val response = authApi.login(LoginRequest(username, password))
            val user = User(
                id = response.user.id,
                username = response.user.username,
                email = response.user.email,
                isAdmin = response.user.is_admin,
                token = response.token,
            )
            persistSession(user)
            telemetry.onAuthEvent("login", username)
            emit(Result.Success(user))
        } catch (ex: Exception) {
            emit(Result.Error(ex.message ?: "Login failed"))
        }
    }

    override fun register(
        username: String,
        email: String,
        password: String,
    ): Flow<Result<User>> = flow {
        emit(Result.Loading)
        try {
            authApi.register(RegisterRequest(username, email, password))
            val response = authApi.login(LoginRequest(username, password))
            val user = User(
                id = response.user.id,
                username = response.user.username,
                email = response.user.email,
                isAdmin = response.user.is_admin,
                token = response.token,
            )
            persistSession(user)
            telemetry.onAuthEvent("register", username)
            emit(Result.Success(user))
        } catch (ex: Exception) {
            emit(Result.Error(ex.message ?: "Registration failed"))
        }
    }

    override suspend fun continueOffline() {
        tokenStore.setGuest("Guest")
        userDao.clear()
        telemetry.onAuthEvent("guest", "Guest")
    }

    override suspend fun logout() {
        tokenStore.clear()
        userDao.clear()
        telemetry.onAuthEvent("logout", null)
    }

    private suspend fun persistSession(user: User) {
        tokenStore.setSession(user.token, user.username)
        userDao.upsert(
            UserEntity(
                id = user.id,
                username = user.username,
                email = user.email,
                token = user.token,
            ),
        )
    }
}
