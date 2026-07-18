package com.example.geolocation.util

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.geolocation.data.repository.MemoryRepository
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject

@HiltWorker
class MemorySyncWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val memoryRepository: MemoryRepository,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val pending = memoryRepository.pendingSync()
        if (pending.isEmpty()) return Result.success()
        var failures = 0
        for (mem in pending) {
            val r = memoryRepository.syncOne(mem)
            if (r.isFailure) failures++
        }
        return when {
            failures == 0 -> Result.success()
            failures < pending.size -> Result.retry()
            else -> Result.retry()
        }
    }
}
