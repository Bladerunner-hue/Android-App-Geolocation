package com.example.geolocation.domain

import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Rotates session id after ~60 minutes of inactivity for leakage-safe splits.
 */
@Singleton
class TrainModeSession @Inject constructor() {
    private var sessionId: String = UUID.randomUUID().toString()
    private var lastActivityMs: Long = System.currentTimeMillis()

    @Synchronized
    fun current(nowMs: Long = System.currentTimeMillis()): String {
        if (nowMs - lastActivityMs > SESSION_GAP_MS) {
            sessionId = UUID.randomUUID().toString()
        }
        lastActivityMs = nowMs
        return sessionId
    }

    companion object {
        const val SESSION_GAP_MS = 60L * 60L * 1000L
    }
}
