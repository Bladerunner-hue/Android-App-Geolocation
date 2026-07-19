package com.example.geolocation.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.example.geolocation.data.local.entity.MemoryTrainingLabelEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface MemoryTrainingLabelDao {
    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insert(label: MemoryTrainingLabelEntity)

    @Update
    suspend fun update(label: MemoryTrainingLabelEntity)

    @Query("SELECT * FROM memory_training_labels WHERE memoryId = :memoryId ORDER BY labelledAtEpochMillis DESC")
    suspend fun forMemory(memoryId: String): List<MemoryTrainingLabelEntity>

    @Query(
        """
        SELECT * FROM memory_training_labels
        WHERE consentForTraining = 1
        ORDER BY labelledAtEpochMillis ASC
        """,
    )
    suspend fun exportableForTraining(): List<MemoryTrainingLabelEntity>

    @Query(
        """
        SELECT * FROM memory_training_labels
        WHERE syncStatus = 'pending' AND consentForCloud = 1
        ORDER BY labelledAtEpochMillis ASC
        """,
    )
    suspend fun pendingCloudSync(): List<MemoryTrainingLabelEntity>

    @Query("SELECT COUNT(*) FROM memory_training_labels WHERE consentForTraining = 1 AND primaryVibe = :vibe")
    suspend fun countForVibe(vibe: String): Int

    @Query("SELECT * FROM memory_training_labels ORDER BY labelledAtEpochMillis DESC")
    fun observeAll(): Flow<List<MemoryTrainingLabelEntity>>
}
