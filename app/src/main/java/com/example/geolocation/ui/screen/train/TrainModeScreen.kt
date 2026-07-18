package com.example.geolocation.ui.screen.train

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

val VIBE_OPTIONS = listOf(
    "serene", "energetic", "chaotic", "nostalgic", "tense", "social", "contemplative",
)

data class TrainModeUiState(
    val memoryId: String = "",
    val primaryVibe: String? = null,
    val valence: Float = 0f,
    val arousal: Float = 3f,
    val confidence: Float = 2f,
    val consentTraining: Boolean = true,
    val consentCloud: Boolean = false,
    val note: String = "",
    val message: String? = null,
    /** Model prediction is intentionally hidden until after label submit. */
    val predictionUnlocked: Boolean = false,
    val modelPrediction: String? = null,
)

/**
 * Label FIRST, then optionally reveal model prediction.
 * Prevents anchoring contamination of the training set.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainModeScreen(
    state: TrainModeUiState,
    onSelectVibe: (String) -> Unit,
    onValence: (Float) -> Unit,
    onArousal: (Float) -> Unit,
    onConfidence: (Float) -> Unit,
    onConsentTraining: (Boolean) -> Unit,
    onConsentCloud: (Boolean) -> Unit,
    onNote: (String) -> Unit,
    onSubmitLabel: () -> Unit,
    onBack: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Train Mode") },
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
            Text("Label how this place felt to you before seeing any model guess.")
            Spacer(Modifier.height(12.dp))
            Text("Primary vibe")
            LazyRow {
                items(VIBE_OPTIONS) { v ->
                    FilterChip(
                        selected = state.primaryVibe == v,
                        onClick = { onSelectVibe(v) },
                        label = { Text(v) },
                        modifier = Modifier.padding(end = 4.dp),
                    )
                }
            }
            Spacer(Modifier.height(12.dp))
            Text("Valence (−2..2): ${state.valence.toInt()}")
            Slider(
                value = state.valence,
                onValueChange = onValence,
                valueRange = -2f..2f,
                steps = 3,
            )
            Text("Arousal (1..5): ${state.arousal.toInt()}")
            Slider(
                value = state.arousal,
                onValueChange = onArousal,
                valueRange = 1f..5f,
                steps = 3,
            )
            Text("Your confidence (1..3): ${state.confidence.toInt()}")
            Slider(
                value = state.confidence,
                onValueChange = onConfidence,
                valueRange = 1f..3f,
                steps = 1,
            )
            RowSwitch("Consent for training", state.consentTraining, onConsentTraining)
            RowSwitch("Consent for cloud upload", state.consentCloud, onConsentCloud)
            OutlinedTextField(
                value = state.note,
                onValueChange = onNote,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Optional note") },
            )
            Spacer(Modifier.height(12.dp))
            Button(
                onClick = onSubmitLabel,
                enabled = state.primaryVibe != null && !state.predictionUnlocked,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Save human label")
            }
            if (state.predictionUnlocked) {
                Spacer(Modifier.height(8.dp))
                Text("Model (after your label): ${state.modelPrediction ?: "unavailable"}")
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
