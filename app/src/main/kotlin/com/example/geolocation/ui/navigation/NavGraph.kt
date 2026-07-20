package com.example.geolocation.ui.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
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
import com.example.geolocation.ui.screen.train.TrainModeScreen
import com.example.geolocation.ui.screen.train.TrainModeViewModel

object Destinations {
    const val LOGIN = "login"
    const val HOME = "home"
    const val DASHBOARD = "dashboard"
    const val LOCATION = "location"
    const val JOURNAL = "journal"
    const val CAPTURE = "capture"
    const val PRIVACY = "privacy"
    const val TRAIN = "train/{memoryId}"

    fun train(memoryId: String) = "train/$memoryId"
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
                onEmailChanged = viewModel::onEmailChanged,
                onLogin = { viewModel.login() },
                onRegister = { viewModel.register() },
                onToggleRegisterMode = viewModel::toggleRegisterMode,
                onContinueOffline = viewModel::continueOffline,
                onAuthenticated = {
                    navController.navigate(Destinations.HOME) {
                        popUpTo(Destinations.LOGIN) { inclusive = true }
                    }
                },
                onGuestContinue = {
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
                onNavigateToJournal = { navController.navigate(Destinations.JOURNAL) },
                onLogout = {
                    viewModel.logout()
                    navController.navigate(Destinations.LOGIN) {
                        popUpTo(0)
                    }
                },
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
                onTrain = { memoryId -> navController.navigate(Destinations.train(memoryId)) },
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
                onLocationPermissionResult = viewModel::onLocationPermissionResult,
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

        composable(
            route = Destinations.TRAIN,
            arguments = listOf(navArgument("memoryId") { type = NavType.StringType }),
        ) {
            val viewModel: TrainModeViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            TrainModeScreen(
                state = state,
                onSelectVibe = viewModel::onSelectVibe,
                onValence = viewModel::onValence,
                onArousal = viewModel::onArousal,
                onConfidence = viewModel::onConfidence,
                onConsentTraining = viewModel::onConsentTraining,
                onConsentCloud = viewModel::onConsentCloud,
                onNote = viewModel::onNote,
                onSubmitLabel = viewModel::submitLabel,
                onBack = { navController.popBackStack() },
            )
        }
    }
}
