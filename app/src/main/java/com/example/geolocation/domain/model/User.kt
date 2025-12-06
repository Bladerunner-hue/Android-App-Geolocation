package com.example.geolocation.domain.model

data class User(
    val id: Int,
    val username: String,
    val email: String,
    val isAdmin: Boolean = false,
    val token: String = ""
)
