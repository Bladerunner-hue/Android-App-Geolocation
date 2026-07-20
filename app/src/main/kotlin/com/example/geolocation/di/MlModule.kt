package com.example.geolocation.di

import com.example.geolocation.data.ml.EdgeMemoryAnalyzer
import com.example.geolocation.data.ml.FusionV0EdgeAnalyzer
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class MlModule {
    @Binds
    @Singleton
    abstract fun bindEdgeMemoryAnalyzer(impl: FusionV0EdgeAnalyzer): EdgeMemoryAnalyzer
}
