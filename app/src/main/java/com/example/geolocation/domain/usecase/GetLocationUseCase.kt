package com.example.geolocation.domain.usecase

import com.example.geolocation.data.repository.LocationRepository

class GetLocationUseCase(
    private val repository: LocationRepository
) {
    operator fun invoke() = repository.observeLocations()
    suspend fun refresh() = repository.refresh()
}
