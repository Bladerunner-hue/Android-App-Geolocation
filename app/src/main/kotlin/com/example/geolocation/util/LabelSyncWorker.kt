package com.example.geolocation.util

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.geolocation.data.repository.TrainingLabelRepository
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject

@HiltWorker
class LabelSyncWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val trainingLabelRepository: TrainingLabelRepository,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val pending = trainingLabelRepository.pendingCloudSync()
        if (pending.isEmpty()) return Result.success()
        var failures = 0
        for (label in pending) {
            if (trainingLabelRepository.syncOne(label).isFailure) failures++
        }
        return if (failures == 0) Result.success() else Result.retry()
    }
}
