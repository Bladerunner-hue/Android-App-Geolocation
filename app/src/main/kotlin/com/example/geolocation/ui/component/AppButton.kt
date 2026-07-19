package com.example.geolocation.ui.component

import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.example.geolocation.ui.theme.PrimaryBlue

@Composable
fun AppButton(
    text: String,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    Button(
        modifier = modifier,
        enabled = enabled,
        colors = ButtonDefaults.buttonColors(containerColor = PrimaryBlue),
        onClick = onClick
    ) {
        Text(text = text)
    }
}
