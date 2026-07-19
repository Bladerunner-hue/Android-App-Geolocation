package com.example.geolocation.util

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SyncScheduler @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun enqueueMemorySync() {
        enqueue(MemorySyncWorker::class.java, MEMORY_UNIQUE)
    }

    fun enqueueLabelSync() {
        enqueue(LabelSyncWorker::class.java, LABEL_UNIQUE)
    }

    private fun enqueue(
        workerClass: Class<out androidx.work.ListenableWorker>,
        name: String,
    ) {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()
        val request = OneTimeWorkRequest.Builder(workerClass)
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            name,
            ExistingWorkPolicy.KEEP,
            request,
        )
    }

    companion object {
        const val MEMORY_UNIQUE = "geojournal_memory_sync"
        const val LABEL_UNIQUE = "geojournal_label_sync"
    }
}
