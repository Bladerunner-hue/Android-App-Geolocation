package com.example.geolocation.data.repository

import com.example.geolocation.data.local.dao.UserDao
import com.example.geolocation.data.local.entity.UserEntity
import com.example.geolocation.data.remote.api.AuthApi
import com.example.geolocation.data.remote.dto.LoginRequest
import com.example.geolocation.domain.model.User
import com.example.geolocation.util.Result
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.update

interface AuthRepository {
    val token: Flow<String?>
    fun login(username: String, password: String): Flow<Result<User>>
    suspend fun logout()
}

@Singleton
class AuthRepositoryImpl @Inject constructor(
    private val authApi: AuthApi,
    private val userDao: UserDao
) : AuthRepository {

    private val tokenState = MutableStateFlow<String?>(null)
    override val token = tokenState.asStateFlow()

    override fun login(username: String, password: String): Flow<Result<User>> = flow {
        emit(Result.Loading)

        try {
            val response = authApi.login(LoginRequest(username, password))
            val user = User(
                id = response.user.id,
                username = response.user.username,
                email = response.user.email,
                isAdmin = response.user.is_admin,
                token = response.token
            )

            userDao.upsert(
                UserEntity(
                    id = user.id,
                    username = user.username,
                    email = user.email,
                    token = user.token
                )
            )
            tokenState.update { response.token }
            emit(Result.Success(user))
        } catch (ex: Exception) {
            emit(Result.Error(ex.message ?: "Login failed"))
        }
    }

    override suspend fun logout() {
        tokenState.update { null }
        userDao.clear()
    }
}
