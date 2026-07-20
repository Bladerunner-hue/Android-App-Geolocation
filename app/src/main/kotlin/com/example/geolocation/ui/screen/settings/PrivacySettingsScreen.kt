package com.example.geolocation.ui.screen.settings

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
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
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PrivacySettingsScreen(
    state: PrivacyUiState,
    onPrivateMode: (Boolean) -> Unit,
    onCloudSync: (Boolean) -> Unit,
    onEnrichment: (Boolean) -> Unit,
    onAudio: (Boolean) -> Unit,
    onExportTrainingBronze: () -> Unit,
    onBack: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Privacy & training") },
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
            Text("Private Mode defaults ON. Cloud never sees Private Mode memories.")
            Spacer(Modifier.height(16.dp))
            ToggleRow("Private Mode", state.privateMode, onPrivateMode)
            ToggleRow(
                "Cloud sync (requires Private Mode OFF)",
                state.cloudSync && !state.privateMode,
                enabled = !state.privateMode,
                onChange = onCloudSync,
            )
            ToggleRow(
                "Backend enrichment (opt-in, requires sync)",
                state.enrichment && state.cloudSync && !state.privateMode,
                enabled = !state.privateMode && state.cloudSync,
                onChange = onEnrichment,
            )
            ToggleRow("Enable ambient audio capture", state.audioCapture, onAudio)
            Spacer(Modifier.height(20.dp))
            Text("Local training export (before backend)")
            Text(
                "Writes bronze_events.jsonl + media zip from Train Mode rows with " +
                    "consent_for_training. No cloud. Feed ml/prepare_fusion_dataset.",
            )
            Spacer(Modifier.height(8.dp))
            Button(
                onClick = onExportTrainingBronze,
                enabled = !state.exportBusy,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (state.exportBusy) "Exporting…" else "Export consented training bronze")
            }
            state.exportMessage?.let {
                Spacer(Modifier.height(8.dp))
                Text(it)
            }
            Spacer(Modifier.height(16.dp))
            Text(
                "An API failure never silently sends media to a cloud LLM. " +
                    "Microphone is requested only when audio is enabled.",
            )
        }
    }
}

@Composable
private fun ToggleRow(
    label: String,
    checked: Boolean,
    onChange: (Boolean) -> Unit,
    enabled: Boolean = true,
) {
    Row(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChange, enabled = enabled)
    }
}
