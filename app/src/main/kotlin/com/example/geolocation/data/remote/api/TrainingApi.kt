package com.example.geolocation.data.remote.api

import com.example.geolocation.data.remote.dto.TrainingLabelDto
import com.example.geolocation.data.remote.dto.TrainingLabelRequest
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

interface TrainingApi {
    @POST("api/training/labels")
    suspend fun createLabel(@Body body: TrainingLabelRequest): TrainingLabelDto

    @GET("api/training/labels/{id}")
    suspend fun getLabel(@Path("id") id: String): TrainingLabelDto

    @GET("api/training/labels")
    suspend fun listForMemory(
        @Query("memory_id") memoryId: String,
        @Query("training_only") trainingOnly: Boolean = false,
    ): List<TrainingLabelDto>
}
