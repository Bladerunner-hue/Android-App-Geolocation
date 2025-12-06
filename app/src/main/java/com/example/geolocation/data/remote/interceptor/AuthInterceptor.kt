package com.example.geolocation.data.remote.interceptor

import android.content.Context
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.Response
import javax.inject.Inject

private val Context.dataStore by preferencesDataStore(name = "auth_prefs")

class AuthInterceptor @Inject constructor(
    @ApplicationContext private val context: Context
) : Interceptor {
    
    companion object {
        val TOKEN_KEY = stringPreferencesKey("auth_token")
    }
    
    override fun intercept(chain: Interceptor.Chain): Response {
        val token = runBlocking {
            context.dataStore.data.map { prefs -> prefs[TOKEN_KEY] }.first()
        }
        
        val requestBuilder = chain.request().newBuilder()
            .addHeader("Accept", "application/json")
        
        if (!token.isNullOrBlank()) {
            requestBuilder.addHeader("Authorization", "Bearer $token")
        }
        
        return chain.proceed(requestBuilder.build())
    }
}
