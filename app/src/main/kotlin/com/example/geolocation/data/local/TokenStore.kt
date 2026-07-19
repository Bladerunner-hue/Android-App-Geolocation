package com.example.geolocation.data.local

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.sessionDataStore by preferencesDataStore(name = "session_prefs")

/**
 * Single source of truth for JWT (and optional username).
 * AuthInterceptor + AuthRepository both use this — never a second token store.
 */
@Singleton
class TokenStore @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val tokenKey = stringPreferencesKey("auth_token")
    private val usernameKey = stringPreferencesKey("username")

    val token: Flow<String?> = context.sessionDataStore.data.map { it[tokenKey] }
    val username: Flow<String?> = context.sessionDataStore.data.map { it[usernameKey] }

    suspend fun currentToken(): String? = token.first()

    suspend fun setSession(token: String, username: String) {
        context.sessionDataStore.edit { prefs ->
            prefs[tokenKey] = token
            prefs[usernameKey] = username
        }
    }

    suspend fun setGuest(username: String = "Guest") {
        context.sessionDataStore.edit { prefs ->
            prefs.remove(tokenKey)
            prefs[usernameKey] = username
        }
    }

    suspend fun clear() {
        context.sessionDataStore.edit { it.clear() }
    }
}
