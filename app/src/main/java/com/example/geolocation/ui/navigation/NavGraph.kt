package com.example.geolocation.ui.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.geolocation.ui.screen.auth.LoginScreen
import com.example.geolocation.ui.screen.auth.LoginViewModel
import com.example.geolocation.ui.screen.dashboard.DashboardScreen
import com.example.geolocation.ui.screen.dashboard.DashboardViewModel
import com.example.geolocation.ui.screen.home.HomeScreen
import com.example.geolocation.ui.screen.home.HomeViewModel
import com.example.geolocation.ui.screen.location.LocationScreen
import com.example.geolocation.ui.screen.location.LocationViewModel
import dagger.hilt.android.lifecycle.HiltViewModel
import androidx.hilt.navigation.compose.hiltViewModel

object Destinations {
    const val LOGIN = "login"
    const val HOME = "home"
    const val DASHBOARD = "dashboard"
    const val LOCATION = "location"
}

@Composable
fun NavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = Destinations.LOGIN
    ) {
        composable(Destinations.LOGIN) {
            val viewModel: LoginViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            LoginScreen(
                state = state,
                onUsernameChanged = viewModel::onUsernameChanged,
                onPasswordChanged = viewModel::onPasswordChanged,
                onLogin = { viewModel.login() },
                onNavigateToRegister = { /* registration flow placeholder */ },
                onAuthenticated = {
                    navController.navigate(Destinations.HOME) {
                        popUpTo(Destinations.LOGIN) { inclusive = true }
                    }
                },
                onErrorDismiss = viewModel::clearError
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
                }
            )
        }

        composable(Destinations.DASHBOARD) {
            val viewModel: DashboardViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            DashboardScreen(
                state = state,
                onBack = { navController.popBackStack() },
                onLocation = { navController.navigate(Destinations.LOCATION) }
            )
        }

        composable(Destinations.LOCATION) {
            val viewModel: LocationViewModel = hiltViewModel()
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            LocationScreen(
                state = state,
                onRefresh = { viewModel.refreshLocation() },
                onBack = { navController.popBackStack() },
                onPermissionGranted = { viewModel.onPermissionGranted() }
            )
        }
    }
}
