package com.example.geolocation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.compose.rememberNavController
import com.example.geolocation.data.telemetry.HiddenTelemetryCollector
import com.example.geolocation.ui.navigation.NavGraph
import com.example.geolocation.ui.theme.GeolocationTheme
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject lateinit var telemetry: HiddenTelemetryCollector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            GeoTrackApp(telemetry = telemetry)
        }
    }
}

@Composable
private fun GeoTrackApp(telemetry: HiddenTelemetryCollector) {
    val navController = rememberNavController()
    GeolocationTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            NavGraph(navController = navController, telemetry = telemetry)
        }
    }
}
