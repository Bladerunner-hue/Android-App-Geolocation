package com.example.geolocation.data.remote.dto

data class LocationResponse(
    val id: Int,
    val latitude: Double,
    val longitude: Double,
    val label: String?,
    val recorded_at: String
)
