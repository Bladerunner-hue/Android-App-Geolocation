package com.example.geolocation.di

import android.content.Context
import androidx.room.Room
import com.example.geolocation.data.local.AppDatabase
import com.example.geolocation.data.local.dao.MemoryDao
import com.example.geolocation.data.local.dao.MemoryTrainingLabelDao
import com.example.geolocation.data.local.dao.UserDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AppDatabase =
        Room.databaseBuilder(context, AppDatabase::class.java, "geojournal.db")
            .addMigrations(AppDatabase.MIGRATION_1_2, AppDatabase.MIGRATION_2_3)
            .fallbackToDestructiveMigration()
            .build()

    @Provides
    fun provideUserDao(database: AppDatabase): UserDao = database.userDao()

    @Provides
    fun provideMemoryDao(database: AppDatabase): MemoryDao = database.memoryDao()

    @Provides
    fun provideMemoryTrainingLabelDao(database: AppDatabase): MemoryTrainingLabelDao =
        database.memoryTrainingLabelDao()
}
