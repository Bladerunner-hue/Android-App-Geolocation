package com.example.geolocation.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.example.geolocation.data.local.entity.MemoryEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface MemoryDao {
    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insert(memory: MemoryEntity): Long

    @Update
    suspend fun update(memory: MemoryEntity)

    @Query("SELECT * FROM memories ORDER BY capturedAtMs DESC")
    fun observeAll(): Flow<List<MemoryEntity>>

    @Query("SELECT * FROM memories ORDER BY capturedAtMs DESC")
    suspend fun listAll(): List<MemoryEntity>

    @Query("SELECT * FROM memories WHERE id = :id LIMIT 1")
    suspend fun getById(id: Long): MemoryEntity?

    @Query("SELECT * FROM memories WHERE clientUuid = :uuid LIMIT 1")
    suspend fun getByClientUuid(uuid: String): MemoryEntity?

    @Query(
        """
        SELECT * FROM memories
        WHERE (:q = '')
           OR lower(ifnull(caption, '')) LIKE '%' || lower(:q) || '%'
           OR lower(ifnull(vibeLabel, '')) LIKE '%' || lower(:q) || '%'
        ORDER BY capturedAtMs DESC
        LIMIT :limit
        """,
    )
    suspend fun search(q: String, limit: Int = 50): List<MemoryEntity>

    @Query(
        """
        SELECT * FROM memories
        WHERE syncStatus = 'pending'
          AND privateMode = 0
          AND cloudSyncEnabled = 1
        ORDER BY capturedAtMs ASC
        LIMIT :limit
        """,
    )
    suspend fun pendingSync(limit: Int = 20): List<MemoryEntity>

    @Query("DELETE FROM memories WHERE id = :id")
    suspend fun deleteById(id: Long)
}
