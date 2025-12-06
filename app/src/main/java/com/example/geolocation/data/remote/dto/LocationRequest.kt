package com.example.geolocation.data.remote.dto

data class LocationRequest(
    val latitude: Double,
    val longitude: Double,
    val label: String? = null
)
