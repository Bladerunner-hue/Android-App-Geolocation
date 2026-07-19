package com.example.geolocation.data.remote.dto

data class LoginResponse(
    val token: String,
    val user: RemoteUser
)

data class RemoteUser(
    val id: Int,
    val username: String,
    val email: String,
    val is_admin: Boolean = false
)
