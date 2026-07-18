package com.example.geolocation.ui.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.geolocation.ui.screen.auth.LoginScreen
import com.example.geolocation.ui.screen.auth.LoginViewModel
import com.example.geolocation.ui.screen.capture.CaptureScreen
import com.example.geolocation.ui.screen.capture.CaptureViewModel
import com.example.geolocation.ui.screen.dashboard.DashboardScreen
import com.example.geolocation.ui.screen.dashboard.DashboardViewModel
import com.example.geolocation.ui.screen.home.HomeScreen
import com.example.geolocation.ui.screen.home.HomeViewModel
import com.example.geolocation.ui.screen.journal.JournalScreen
import com.example.geolocation.ui.screen.journal.JournalViewModel
import com.example.geolocation.ui.screen.location.LocationScreen
import com.example.geolocation.ui.screen.location.LocationViewModel
import com.example.geolocation.ui.screen.settings.PrivacySettingsScreen
import com.example.geolocation.ui.screen.settings.PrivacySettingsViewModel

object Destinations {
    const val LOGIN = "login"
    const val HOME = "home"
    const val DASHBOARD = "dashboard"
    const val LOCATION = "location"
    const val JOURNAL = "journal"
    const val CAPTURE = "capture"
    const val PRIVACY = "privacy"
}

@Composable
fun NavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = Destinations.LOGIN,
    ) {
        composable(Destinations.LOGIN) {
            val viewModel: LoginViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            LoginScreen(
                state = state,
                onUsernameChanged = viewModel::onUsernameChanged,
                onPasswordChanged = viewModel::onPasswordChanged,
                onLogin = { viewModel.login() },
                onNavigateToRegister = { },
                onAuthenticated = {
                    navController.navigate(Destinations.HOME) {
                        popUpTo(Destinations.LOGIN) { inclusive = true }
                    }
                },
                onErrorDismiss = viewModel::clearError,
            )
        }

        composable(Destinations.HOME) {
            val viewModel: HomeViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            HomeScreen(
                state = state,
                onNavigateToDashboard = { navController.navigate(Destinations.DASHBOARD) },
                onNavigateToLocation = { navController.navigate(Destinations.LOCATION) },
                onLogout = {
                    navController.navigate(Destinations.LOGIN) {
                        popUpTo(0)
                    }
                },
                onNavigateToJournal = { navController.navigate(Destinations.JOURNAL) },
            )
        }

        composable(Destinations.DASHBOARD) {
            val viewModel: DashboardViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            DashboardScreen(
                state = state,
                onBack = { navController.popBackStack() },
                onLocation = { navController.navigate(Destinations.LOCATION) },
            )
        }

        composable(Destinations.LOCATION) {
            val viewModel: LocationViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            LocationScreen(
                state = state,
                onRefresh = { viewModel.refreshLocation() },
                onBack = { navController.popBackStack() },
                onPermissionGranted = { viewModel.onPermissionGranted() },
            )
        }

        composable(Destinations.JOURNAL) {
            val viewModel: JournalViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            JournalScreen(
                state = state,
                onQueryChange = viewModel::onQueryChange,
                onSearch = viewModel::search,
                onCapture = { navController.navigate(Destinations.CAPTURE) },
                onSettings = { navController.navigate(Destinations.PRIVACY) },
                onBack = { navController.popBackStack() },
            )
        }

        composable(Destinations.CAPTURE) {
            val viewModel: CaptureViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            LaunchedEffect(state.done) {
                if (state.done) navController.popBackStack()
            }
            CaptureScreen(
                state = state,
                onCaptionChange = viewModel::onCaptionChange,
                onToggleAudio = viewModel::onToggleAudio,
                onUseLocation = viewModel::onUseLocation,
                onPhotoReady = viewModel::onPhotoReady,
                onSave = viewModel::save,
                onBack = { navController.popBackStack() },
            )
        }

        composable(Destinations.PRIVACY) {
            val viewModel: PrivacySettingsViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            PrivacySettingsScreen(
                state = state,
                onPrivateMode = viewModel::setPrivateMode,
                onCloudSync = viewModel::setCloudSync,
                onEnrichment = viewModel::setEnrichment,
                onAudio = viewModel::setAudio,
                onBack = { navController.popBackStack() },
            )
        }
    }
}
