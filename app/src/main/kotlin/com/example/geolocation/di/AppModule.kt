package com.example.geolocation.di

import android.content.Context
import com.example.geolocation.data.local.TokenStore
import com.example.geolocation.data.local.dao.UserDao
import com.example.geolocation.data.remote.api.AuthApi
import com.example.geolocation.data.repository.AuthRepository
import com.example.geolocation.data.repository.AuthRepositoryImpl
import com.example.geolocation.data.repository.LocationRepository
import com.example.geolocation.data.repository.LocationRepositoryImpl
import com.example.geolocation.domain.usecase.GetLocationUseCase
import com.example.geolocation.domain.usecase.LoginUseCase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideAuthRepository(
        authApi: AuthApi,
        userDao: UserDao,
        tokenStore: TokenStore,
    ): AuthRepository = AuthRepositoryImpl(authApi, userDao, tokenStore)

    @Provides
    @Singleton
    fun provideLocationRepository(
        @ApplicationContext context: Context
    ): LocationRepository = LocationRepositoryImpl(context)

    @Provides
    @Singleton
    fun provideLoginUseCase(authRepository: AuthRepository) = LoginUseCase(authRepository)

    @Provides
    @Singleton
    fun provideGetLocationUseCase(locationRepository: LocationRepository) =
        GetLocationUseCase(locationRepository)
}
