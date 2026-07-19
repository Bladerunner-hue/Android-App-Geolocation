package com.example.geolocation.data.remote.interceptor

import com.example.geolocation.data.local.TokenStore
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.Response

/**
 * Attaches Bearer JWT from [TokenStore] only.
 * Guest / offline sessions have no token → no Authorization header.
 */
@Singleton
class AuthInterceptor @Inject constructor(
    private val tokenStore: TokenStore,
) : Interceptor {

    override fun intercept(chain: Interceptor.Chain): Response {
        val token = runBlocking { tokenStore.currentToken() }
        val builder = chain.request().newBuilder()
            .header("Accept", "application/json")
        if (!token.isNullOrBlank()) {
            builder.header("Authorization", "Bearer $token")
        }
        return chain.proceed(builder.build())
    }
}
