package com.example.geolocation.domain.model

data class GeoLocation(
    val latitude: Double,
    val longitude: Double,
    val label: String = "",
    val timestamp: Long = System.currentTimeMillis()
)
