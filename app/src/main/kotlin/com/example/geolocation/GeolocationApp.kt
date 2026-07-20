package com.example.geolocation

import android.app.Application
import androidx.hilt.work.HiltWorkerFactory
import androidx.work.Configuration
import com.example.geolocation.data.telemetry.HiddenTelemetryCollector
import com.example.geolocation.data.telemetry.TelemetryPipelineFeeder
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject

@HiltAndroidApp
class GeolocationApp : Application(), Configuration.Provider {

    @Inject lateinit var workerFactory: HiltWorkerFactory
    @Inject lateinit var telemetryCollector: HiddenTelemetryCollector
    @Inject lateinit var telemetryFeeder: TelemetryPipelineFeeder

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setWorkerFactory(workerFactory)
            .build()

    override fun onCreate() {
        super.onCreate()
        // Initialize hidden telemetry on app start
        telemetryCollector.onAppStart()
        // Schedule periodic telemetry uploads
        telemetryFeeder.schedulePeriodic()
    }
}
