package com.example.geolocation.data.remote.api

import com.example.geolocation.data.remote.dto.MemoryDto
import com.example.geolocation.data.remote.dto.MemorySearchDto
import com.example.geolocation.data.remote.dto.VibeProfileDto
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import retrofit2.http.Query

interface MemoryApi {
    @Multipart
    @POST("api/memories/analyze")
    suspend fun analyze(
        @Part("client_uuid") clientUuid: RequestBody,
        @Part("private_mode") privateMode: RequestBody,
        @Part("caption") caption: RequestBody?,
        @Part("latitude") latitude: RequestBody?,
        @Part("longitude") longitude: RequestBody?,
        @Part("on_device_vibe") vibe: RequestBody?,
        @Part("on_device_confidence") confidence: RequestBody?,
        @Part("request_enrichment") enrichment: RequestBody,
        @Part photo: MultipartBody.Part?,
        @Part audio: MultipartBody.Part?,
    ): MemoryDto

    @GET("api/memories/search")
    suspend fun search(@Query("q") q: String, @Query("limit") limit: Int = 20): MemorySearchDto

    @GET("api/memories/{id}")
    suspend fun get(@Path("id") id: Long): MemoryDto

    @GET("api/user/vibe-profile")
    suspend fun vibeProfile(): VibeProfileDto
}
