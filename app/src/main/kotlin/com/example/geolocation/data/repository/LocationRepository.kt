package com.example.geolocation.data.repository

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Geocoder
import android.os.Looper
import androidx.core.content.ContextCompat
import com.example.geolocation.domain.model.GeoLocation
import com.example.geolocation.util.Result
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.tasks.await

interface LocationRepository {
    fun observeLocations(): Flow<Result<GeoLocation>>
    suspend fun refresh(): GeoLocation
    fun hasLocationPermission(): Boolean
}

@Singleton
class LocationRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context
) : LocationRepository {

    private val fusedLocationClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val geocoder: Geocoder by lazy {
        Geocoder(context, Locale.getDefault())
    }

    override fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun observeLocations(): Flow<Result<GeoLocation>> = callbackFlow {
        if (!hasLocationPermission()) {
            trySend(Result.Error("Location permission not granted"))
            close()
            return@callbackFlow
        }

        trySend(Result.Loading)

        val locationRequest = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            5000L // Update every 5 seconds
        ).setMinUpdateIntervalMillis(2000L).build()

        val callback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                result.lastLocation?.let { location ->
                    val label = getAddressFromLocation(location.latitude, location.longitude)
                    val geoLocation = GeoLocation(
                        latitude = location.latitude,
                        longitude = location.longitude,
                        label = label
                    )
                    trySend(Result.Success(geoLocation))
                }
            }
        }

        try {
            fusedLocationClient.requestLocationUpdates(
                locationRequest,
                callback,
                Looper.getMainLooper()
            )
        } catch (e: SecurityException) {
            trySend(Result.Error("Location permission denied"))
            close()
        }

        awaitClose {
            fusedLocationClient.removeLocationUpdates(callback)
        }
    }

    override suspend fun refresh(): GeoLocation {
        return try {
            if (!hasLocationPermission()) {
                return GeoLocation(0.0, 0.0, "Permission denied")
            }
            
            val location = fusedLocationClient.lastLocation.await()
            if (location != null) {
                val label = getAddressFromLocation(location.latitude, location.longitude)
                GeoLocation(location.latitude, location.longitude, label)
            } else {
                GeoLocation(0.0, 0.0, "Location unavailable")
            }
        } catch (e: SecurityException) {
            GeoLocation(0.0, 0.0, "Permission denied")
        } catch (e: Exception) {
            GeoLocation(0.0, 0.0, "Error: ${e.message}")
        }
    }

    private fun getAddressFromLocation(latitude: Double, longitude: Double): String {
        return try {
            @Suppress("DEPRECATION")
            val addresses = geocoder.getFromLocation(latitude, longitude, 1)
            if (!addresses.isNullOrEmpty()) {
                val address = addresses[0]
                listOfNotNull(
                    address.locality,
                    address.adminArea,
                    address.countryName
                ).joinToString(", ")
            } else {
                "Unknown location"
            }
        } catch (e: Exception) {
            "Lat: ${"%.4f".format(latitude)}, Lng: ${"%.4f".format(longitude)}"
        }
    }
}
