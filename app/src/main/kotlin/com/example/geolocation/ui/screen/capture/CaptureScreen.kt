package com.example.geolocation.ui.screen.capture

import android.Manifest
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CaptureScreen(
    state: CaptureUiState,
    onCaptionChange: (String) -> Unit,
    onToggleAudio: (Boolean) -> Unit,
    onUseLocation: (Boolean) -> Unit,
    onLocationPermissionResult: (Boolean) -> Unit,
    onPhotoReady: (File) -> Unit,
    onSave: () -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    var photoFile by remember { mutableStateOf<File?>(null) }

    val takePicture = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture(),
    ) { ok ->
        if (ok) photoFile?.let(onPhotoReady)
    }

    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            val dir = File(context.filesDir, "captures").apply { mkdirs() }
            val f = File(dir, "photo_${System.currentTimeMillis()}.jpg")
            photoFile = f
            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                f,
            )
            takePicture.launch(uri)
        }
    }

    val micPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { /* requested when audio enabled */ }

    val locationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        val granted = result[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
            result[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        onLocationPermissionResult(granted)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Capture memory") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
        ) {
            Button(
                onClick = { cameraPermission.launch(Manifest.permission.CAMERA) },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (state.photoPath != null) "Retake photo" else "Take photo")
            }
            Spacer(Modifier.height(8.dp))
            Text("Photo: ${state.photoPath ?: "none"}")
            Spacer(Modifier.height(12.dp))
            OutlinedTextField(
                value = state.caption,
                onValueChange = onCaptionChange,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Caption (optional)") },
            )
            Spacer(Modifier.height(12.dp))
            RowSwitch("Record ambient audio (≤10s)", state.recordAudio) {
                if (it) micPermission.launch(Manifest.permission.RECORD_AUDIO)
                onToggleAudio(it)
            }
            RowSwitch("Attach location", state.useLocation) { want ->
                if (want && !state.locationPermissionGranted) {
                    locationPermission.launch(
                        arrayOf(
                            Manifest.permission.ACCESS_FINE_LOCATION,
                            Manifest.permission.ACCESS_COARSE_LOCATION,
                        ),
                    )
                }
                onUseLocation(want)
            }
            Text("Location: ${state.locationStatus}")
            Spacer(Modifier.height(8.dp))
            Text(
                "Private Mode is controlled in Privacy settings. " +
                    "Saves go to Room first; network only if sync is enabled.",
            )
            Spacer(Modifier.height(16.dp))
            Button(
                onClick = onSave,
                enabled = !state.saving,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (state.saving) "Saving…" else "Save to journal")
            }
            state.message?.let {
                Spacer(Modifier.height(8.dp))
                Text(it)
            }
        }
    }
}

@Composable
private fun RowSwitch(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    androidx.compose.foundation.layout.Row(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = androidx.compose.foundation.layout.Arrangement.SpaceBetween,
    ) {
        Text(label)
        Switch(checked = checked, onCheckedChange = onChange)
    }
}
