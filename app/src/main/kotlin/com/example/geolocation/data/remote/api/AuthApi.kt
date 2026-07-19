package com.example.geolocation.data.remote.api

import com.example.geolocation.data.remote.dto.LoginRequest
import com.example.geolocation.data.remote.dto.LoginResponse
import com.example.geolocation.data.remote.dto.LocationRequest
import com.example.geolocation.data.remote.dto.LocationResponse
import com.example.geolocation.data.remote.dto.RegisterRequest
import com.example.geolocation.data.remote.dto.RemoteUser
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query

interface AuthApi {
    @POST("api/auth/register")
    suspend fun register(@Body request: RegisterRequest): RemoteUser

    @POST("api/auth/login")
    suspend fun login(@Body request: LoginRequest): LoginResponse

    @POST("api/location")
    suspend fun saveLocation(@Body request: LocationRequest): LocationResponse

    @GET("api/location/history")
    suspend fun getLocationHistory(@Query("limit") limit: Int = 20): List<LocationResponse>
}
