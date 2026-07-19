package com.example.geolocation.ui.screen.auth

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.example.geolocation.ui.component.AppButton
import com.example.geolocation.ui.component.ErrorDialog
import com.example.geolocation.ui.component.LoadingIndicator
import com.example.geolocation.ui.theme.AccentGreen
import com.example.geolocation.ui.theme.PrimaryBlue

@Composable
fun LoginScreen(
    state: LoginUiState,
    onUsernameChanged: (String) -> Unit,
    onPasswordChanged: (String) -> Unit,
    onEmailChanged: (String) -> Unit,
    onLogin: () -> Unit,
    onRegister: () -> Unit,
    onToggleRegisterMode: () -> Unit,
    onContinueOffline: () -> Unit,
    onAuthenticated: () -> Unit,
    onGuestContinue: () -> Unit,
    onErrorDismiss: () -> Unit,
) {
    LaunchedEffect(state.isAuthenticated) {
        if (state.isAuthenticated) onAuthenticated()
    }
    LaunchedEffect(state.continueAsGuest) {
        if (state.continueAsGuest) onGuestContinue()
    }

    val gradient = Brush.verticalGradient(
        colors = listOf(PrimaryBlue, AccentGreen.copy(alpha = 0.35f)),
        startY = 0f,
        endY = 1200f,
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(gradient),
        contentAlignment = Alignment.Center,
    ) {
        Surface(
            modifier = Modifier
                .padding(horizontal = 24.dp)
                .shadow(12.dp, RoundedCornerShape(16.dp)),
            color = MaterialTheme.colorScheme.background,
            shape = RoundedCornerShape(16.dp),
        ) {
            Column(
                modifier = Modifier
                    .padding(24.dp)
                    .fillMaxWidth(),
                verticalArrangement = Arrangement.Center,
            ) {
                Text(
                    text = "GeoJournal",
                    style = MaterialTheme.typography.displayLarge,
                    color = PrimaryBlue,
                )
                Text(
                    text = if (state.isRegisterMode) {
                        "Create an account for optional cloud sync"
                    } else {
                        "Private journal first · cloud is optional"
                    },
                    style = MaterialTheme.typography.bodyLarge,
                    color = Color.Gray,
                )
                Spacer(modifier = Modifier.height(24.dp))

                TextField(
                    value = state.username,
                    onValueChange = onUsernameChanged,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Username") },
                    singleLine = true,
                )
                Spacer(modifier = Modifier.height(12.dp))

                if (state.isRegisterMode) {
                    TextField(
                        value = state.email,
                        onValueChange = onEmailChanged,
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Email") },
                        singleLine = true,
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                }

                TextField(
                    value = state.password,
                    onValueChange = onPasswordChanged,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Password") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                )

                Spacer(modifier = Modifier.height(20.dp))
                AppButton(
                    text = when {
                        state.isLoading && state.isRegisterMode -> "Creating…"
                        state.isLoading -> "Signing in…"
                        state.isRegisterMode -> "Create account"
                        else -> "Login"
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    enabled = !state.isLoading &&
                        state.username.isNotBlank() &&
                        state.password.isNotBlank() &&
                        (!state.isRegisterMode || state.email.isNotBlank()),
                    onClick = {
                        if (state.isRegisterMode) onRegister() else onLogin()
                    },
                )

                if (state.isLoading) {
                    LoadingIndicator()
                }

                TextButton(
                    onClick = onToggleRegisterMode,
                    modifier = Modifier.align(Alignment.CenterHorizontally),
                ) {
                    Text(
                        text = if (state.isRegisterMode) {
                            "Have an account? Sign in"
                        } else {
                            "New here? Create an account"
                        },
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }

                TextButton(
                    onClick = onContinueOffline,
                    modifier = Modifier.align(Alignment.CenterHorizontally),
                    enabled = !state.isLoading,
                ) {
                    Text(
                        text = "Continue free offline",
                        fontWeight = FontWeight.SemiBold,
                        color = AccentGreen,
                    )
                }

                Text(
                    text = "Offline uses Room + Train Mode only. No backend credentials required.",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray,
                )
            }
        }
    }

    state.error?.let { message ->
        ErrorDialog(message = message, onDismiss = onErrorDismiss)
    }
}
