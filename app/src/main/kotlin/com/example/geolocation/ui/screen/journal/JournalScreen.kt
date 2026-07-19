package com.example.geolocation.ui.screen.journal

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.geolocation.data.local.entity.MemoryEntity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun JournalScreen(
    state: JournalUiState,
    onQueryChange: (String) -> Unit,
    onSearch: () -> Unit,
    onCapture: () -> Unit,
    onSettings: () -> Unit,
    onTrain: (memoryId: String) -> Unit,
    onBack: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("GeoJournal") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = onSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Privacy")
                    }
                },
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = onCapture) {
                Icon(Icons.Default.Add, contentDescription = "Capture memory")
            }
        },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
        ) {
            OutlinedTextField(
                value = state.query,
                onValueChange = onQueryChange,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Search local journal") },
                singleLine = true,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Private Mode ${if (state.privateMode) "ON" else "OFF"} · " +
                    "Analysis: ${state.mlStatus}",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Tap a memory → Train Mode (label before model reveal). Free offline collection.",
                style = MaterialTheme.typography.bodySmall,
            )
            Spacer(Modifier.height(8.dp))
            LazyColumn(
                contentPadding = PaddingValues(bottom = 88.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(state.memories, key = { it.id }) { mem ->
                    MemoryCard(mem = mem, onClick = { onTrain(mem.id.toString()) })
                }
            }
        }
    }
}

@Composable
private fun MemoryCard(mem: MemoryEntity, onClick: () -> Unit) {
    val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                mem.caption?.ifBlank { null } ?: "(no caption)",
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                "vibe=${mem.vibeLabel ?: "—"} · ${mem.analysisStatus} · sync=${mem.syncStatus}",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                fmt.format(Date(mem.capturedAtMs)),
                style = MaterialTheme.typography.labelSmall,
            )
            if (mem.latitude != null && mem.longitude != null) {
                Text(
                    "loc=${"%.4f".format(mem.latitude)}, ${"%.4f".format(mem.longitude)}",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
            Text(
                "Train Mode →",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
            )
        }
    }
}
